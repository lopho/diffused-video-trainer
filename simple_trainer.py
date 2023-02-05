import os
import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from accelerate import Accelerator
from diffusers import DDPMScheduler, UNetPseudo3DConditionModel, AutoencoderKL
from diffusers.optimization import get_scheduler
from transformers import CLIPTokenizer, CLIPTextModel
from tqdm.auto import tqdm
from PIL import Image, ImageOps
from torchvision.transforms.functional import pil_to_tensor, to_pil_image

from einops import rearrange

def pil_to_torch(image, device = 'cpu'):
    return (2 * (pil_to_tensor(image).to(dtype=torch.float32, device=device)) / 255) - 1

def torch_to_pil(x):
    return to_pil_image((x + 1) / 2)

def encode_latents(path, outpath, model):
    files: list[str] = os.listdir(path)
    files.sort()
    os.makedirs(outpath, exist_ok=True)
    vae = AutoencoderKL.from_pretrained(model, subfolder='vae').to('cuda:0')
    for f in tqdm(files):
        im = Image.open(os.path.join(path, f))
        im = ImageOps.fit(im, (512, 512), centering = (0.5, 0.5))
        with torch.inference_mode():
            m = pil_to_torch(im, 'cuda:0').unsqueeze(0)
            m = vae.encode(m).latent_dist
            torch.save({ 'mean': m.mean.squeeze().cpu(), 'std': m.std.squeeze().cpu() }, os.path.join(outpath, os.path.splitext(f)[0] + '.pt'))

def encode_prompts(prompts, outpath, model):
    tokenizer = CLIPTokenizer.from_pretrained(model, subfolder='tokenizer')
    text_encoder = CLIPTextModel.from_pretrained(model, subfolder='text_encoder').to('cuda:0')
    for i, prompt in enumerate(prompts):
        with torch.inference_mode():
            tokens = tokenizer(
                    [ prompt ],
                    truncation = True,
                    return_overflowing_tokens = False,
                    padding = 'max_length',
                    return_tensors = 'pt'
            ).input_ids.to('cuda:0')
            y = text_encoder(input_ids = tokens).last_hidden_state
            torch.save(y.cpu(), os.path.join(outpath, str(i).zfill(4) + '.pt' ))

def load_dataset(latent_path, prompt_path, batch_size, frames):
    files: list[str] = os.listdir(latent_path)
    files.sort()
    assert len(files) >= batch_size * frames
    # just make one batch for testing
    files = files[:batch_size * frames]
    prompt: torch.Tensor = torch.load(prompt_path)
    prompt = prompt.pin_memory('cuda:0')
    latents: list[torch.Tensor] = []
    for f in tqdm(files):
        l: dict[str, torch.Tensor] = torch.load(os.path.join(latent_path, f))
        latents.append(l['mean'].squeeze())
    latents: torch.Tensor = torch.stack(latents).unsqueeze(0)
    latents = rearrange(latents, 'b f c h w -> b c f h w').unsqueeze(0)
    return latents.to('cuda:0'), prompt.to('cuda:0')

def main(epochs: int = 10):
    pretrained_model_name_or_path = 'lxj616/make-a-stable-diffusion-video-timelapse'
    learning_rate = 5e-6
    gradient_accumulation_steps = 1
    batch_size = 1
    frames_length = 25
    encoded, prompt = load_dataset('latents', 'prompt.pt', batch_size, frames_length)
    dataset_size = len(encoded)
    lr_warmup_steps = 100
    unfreeze_all = True

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision='bf16',
        log_with='tensorboard',
        logging_dir='logs'
    )
    unet = UNetPseudo3DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder='unet'
    )
    unet.enable_gradient_checkpointing()
    unet.enable_xformers_memory_efficient_attention()

    unet.train()
    if not unfreeze_all:
        unet.requires_grad_(False)
        for name, param in unet.named_parameters():
            if 'temporal_conv' in name:
                param.requires_grad_(True)
        for block in unet.down_blocks:
            if hasattr(block, 'attentions') and block.attentions is not None:
                for attn_block in block.attentions:
                    for transformer_block in attn_block.transformer_blocks:
                        transformer_block.requires_grad_(False)
                        transformer_block.attn_temporal.requires_grad_(True)
                        transformer_block.norm_temporal.requires_grad_(True)
        for block in [unet.mid_block,]:
            if hasattr(block, 'attentions') and block.attentions is not None:
                for attn_block in block.attentions:
                    for transformer_block in attn_block.transformer_blocks:
                        transformer_block.requires_grad_(False)
                        transformer_block.attn_temporal.requires_grad_(True)
                        transformer_block.norm_temporal.requires_grad_(True)
        for block in unet.up_blocks:
            if hasattr(block, 'attentions') and block.attentions is not None:
                for attn_block in block.attentions:
                    for transformer_block in attn_block.transformer_blocks:
                        transformer_block.requires_grad_(False)
                        transformer_block.attn_temporal.requires_grad_(True)
                        transformer_block.norm_temporal.requires_grad_(True)
    params_to_optimize = (
        filter(lambda p: p.requires_grad, unet.parameters()) 
    )
    optimizer_class = torch.optim.AdamW
    optimizer = optimizer_class(
        params_to_optimize,
        lr = learning_rate
    )

    noise_scheduler = DDPMScheduler.from_config(pretrained_model_name_or_path, subfolder="scheduler")

    num_update_steps_per_epoch = math.ceil(dataset_size / gradient_accumulation_steps)
    max_train_steps = epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        'constant_with_warmup',
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    unet, optimizer, lr_scheduler = accelerator.prepare(
        unet, optimizer, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(dataset_size / gradient_accumulation_steps)
    max_train_steps = epochs * num_update_steps_per_epoch

    if accelerator.is_main_process:
        accelerator.init_trackers("video_diffusion")

    progress_bar = tqdm(range(max_train_steps))
    global_step = 0
    tqdm.write(max_train_steps)
    tqdm.write(dataset_size)
    tqdm.write(epochs)

    for epoch in range(epochs):
        for step, batch in enumerate(encoded):
            with accelerator.accumulate(unet):
                latents = batch * 0.18215
                hint_latent = latents[:,:,:1,:,:]
                input_latents = latents[:,:,1:,:,:]
                hint_latent = hint_latent
                input_latents = input_latents
                noise = torch.randn_like(input_latents)
                bsz = input_latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=input_latents.device)
                timesteps = timesteps.long()
                noisy_latents = noise_scheduler.add_noise(input_latents, noise, timesteps)
                encoder_hidden_states = prompt
                mask = torch.zeros([noisy_latents.shape[0], 1, noisy_latents.shape[2], noisy_latents.shape[3], noisy_latents.shape[4]]).to(accelerator.device)
                latent_model_input = torch.cat([noisy_latents, mask, hint_latent.expand(-1,-1,noisy_latents.shape[2],-1,-1)], dim=1).to(accelerator.device)
                with accelerator.autocast():
                    noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = (
                        unet.parameters()
                    )
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
            logs = {"loss": loss.item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= max_train_steps:
                break
        accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet, keep_fp32_wrapper = False)
        torch.save(unet.state_dict(), 'unet.pt')
    accelerator.end_training()
