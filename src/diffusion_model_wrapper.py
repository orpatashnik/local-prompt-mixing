from typing import Optional, List

import numpy as np
import torch
from cv2 import dilate
from diffusers import DDIMScheduler, StableDiffusionPipeline
from tqdm import tqdm

from src.attention_based_segmentation import Segmentor
from src.attention_utils import show_cross_attention
from src.prompt_to_prompt_controllers import DummyController, AttentionStore


def get_stable_diffusion_model(args):
    device = torch.device(f'cuda:{args.gpu_id}') if torch.cuda.is_available() else torch.device('cpu')
    if args.real_image_path != "":
        scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
        ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=args.auth_token, scheduler=scheduler).to(device)
    else:
        ldm_stable = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=args.auth_token).to(device)

    return ldm_stable

def get_stable_diffusion_config(args):
    return {
        "low_resource": args.low_resource,
        "num_diffusion_steps": args.num_diffusion_steps,
        "guidance_scale": args.guidance_scale,
        "max_num_words": args.max_num_words
    }


def generate_original_image(args, ldm_stable, ldm_stable_config, prompts, latent, uncond_embeddings):
    g_cpu = torch.Generator(device=ldm_stable.device).manual_seed(args.seed)
    controller = AttentionStore(ldm_stable_config["low_resource"])
    diffusion_model_wrapper = DiffusionModelWrapper(args, ldm_stable, ldm_stable_config, controller, generator=g_cpu)
    image, x_t, orig_all_latents, _ = diffusion_model_wrapper.forward(prompts,
                                                                      latent=latent,
                                                                      uncond_embeddings=uncond_embeddings)
    orig_mask = Segmentor(controller, prompts, args.num_segments, args.background_segment_threshold, background_nouns=args.background_nouns)\
        .get_background_mask(args.prompt.split(' ').index("{word}") + 1)
    average_attention = controller.get_average_attention()
    return image, x_t, orig_all_latents, orig_mask, average_attention


class DiffusionModelWrapper:
    def __init__(self, args, model, model_config, controller=None, prompt_mixing=None, generator=None):
        self.args = args
        self.model = model
        self.model_config = model_config
        self.controller = controller
        if self.controller is None:
            self.controller = DummyController()
        self.prompt_mixing = prompt_mixing
        self.device = model.device
        self.generator = generator

        self.height = 512
        self.width = 512

        self.diff_step = 0
        self.register_attention_control()


    def diffusion_step(self, latents, context, t, other_context=None):
        if self.model_config["low_resource"]:
            self.uncond_pred = True
            noise_pred_uncond = self.model.unet(latents, t, encoder_hidden_states=(context[0], None))["sample"]
            self.uncond_pred = False
            noise_prediction_text = self.model.unet(latents, t, encoder_hidden_states=(context[1], other_context))["sample"]
        else:
            latents_input = torch.cat([latents] * 2)
            noise_pred = self.model.unet(latents_input, t, encoder_hidden_states=(context, other_context))["sample"]
            noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.model_config["guidance_scale"] * (noise_prediction_text - noise_pred_uncond)
        latents = self.model.scheduler.step(noise_pred, t, latents)["prev_sample"]
        latents = self.controller.step_callback(latents)
        return latents


    def latent2image(self, latents):
        latents = 1 / 0.18215 * latents
        image = self.model.vae.decode(latents)['sample']
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).astype(np.uint8)
        return image


    def init_latent(self, latent, batch_size):
        if latent is None:
            latent = torch.randn(
                (1, self.model.unet.in_channels, self.height // 8, self.width // 8),
                generator=self.generator, device=self.model.device
            )
        latents = latent.expand(batch_size,  self.model.unet.in_channels, self.height // 8, self.width // 8).to(self.device)
        return latent, latents


    def register_attention_control(self):
        def ca_forward(model_self, place_in_unet):
            to_out = model_self.to_out
            if type(to_out) is torch.nn.modules.container.ModuleList:
                to_out = model_self.to_out[0]
            else:
                to_out = model_self.to_out

            def forward(x, context=None, mask=None):
                batch_size, sequence_length, dim = x.shape
                h = model_self.heads
                q = model_self.to_q(x)
                is_cross = context is not None
                context = context if is_cross else (x, None)

                k = model_self.to_k(context[0])
                if is_cross and self.prompt_mixing is not None:
                    v_context = self.prompt_mixing.get_context_for_v(self.diff_step, context[0], context[1])
                    v = model_self.to_v(v_context)
                else:
                    v = model_self.to_v(context[0])

                q = model_self.reshape_heads_to_batch_dim(q)
                k = model_self.reshape_heads_to_batch_dim(k)
                v = model_self.reshape_heads_to_batch_dim(v)

                sim = torch.einsum("b i d, b j d -> b i j", q, k) * model_self.scale

                if mask is not None:
                    mask = mask.reshape(batch_size, -1)
                    max_neg_value = -torch.finfo(sim.dtype).max
                    mask = mask[:, None, :].repeat(h, 1, 1)
                    sim.masked_fill_(~mask, max_neg_value)

                # attention, what we cannot get enough of
                attn = sim.softmax(dim=-1)
                if self.enbale_attn_controller_changes:
                    attn = self.controller(attn, is_cross, place_in_unet)
                
                if is_cross and self.prompt_mixing is not None and context[1] is not None:
                    attn = self.prompt_mixing.get_cross_attn(self, self.diff_step, attn, place_in_unet, batch_size)

                if not is_cross and (not self.model_config["low_resource"] or not self.uncond_pred) and self.prompt_mixing is not None:
                    attn = self.prompt_mixing.get_self_attn(self, self.diff_step, attn, place_in_unet, batch_size)

                out = torch.einsum("b i j, b j d -> b i d", attn, v)
                out = model_self.reshape_batch_dim_to_heads(out)
                return to_out(out)

            return forward

        def register_recr(net_, count, place_in_unet):
            if net_.__class__.__name__ == 'CrossAttention':
                net_.forward = ca_forward(net_, place_in_unet)
                return count + 1
            elif hasattr(net_, 'children'):
                for net__ in net_.children():
                    count = register_recr(net__, count, place_in_unet)
            return count

        cross_att_count = 0
        sub_nets = self.model.unet.named_children()
        for net in sub_nets:
            if "down" in net[0]:
                cross_att_count += register_recr(net[1], 0, "down")
            elif "up" in net[0]:
                cross_att_count += register_recr(net[1], 0, "up")
            elif "mid" in net[0]:
                cross_att_count += register_recr(net[1], 0, "mid")
        self.controller.num_att_layers = cross_att_count


    def get_text_embedding(self, prompt: List[str], max_length=None, truncation=True):
        text_input = self.model.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length if max_length is None else max_length,
            truncation=truncation,
            return_tensors="pt",
        )
        text_embeddings = self.model.text_encoder(text_input.input_ids.to(self.device))[0]
        max_length = text_input.input_ids.shape[-1]
        return text_embeddings, max_length


    @torch.no_grad()
    def forward(self, prompt: List[str], latent: Optional[torch.FloatTensor] = None,
                other_prompt: List[str] = None, post_background = False, orig_all_latents = None, orig_mask = None,
                uncond_embeddings=None, start_time=51, return_type='image'):
        self.enbale_attn_controller_changes = True
        batch_size = len(prompt)

        text_embeddings, max_length = self.get_text_embedding(prompt)
        if uncond_embeddings is None:
            uncond_embeddings_, _ = self.get_text_embedding([""] * batch_size, max_length=max_length, truncation=False)
        else:
            uncond_embeddings_ = None

        other_context = None
        if other_prompt is not None:
            other_text_embeddings, _ = self.get_text_embedding(other_prompt)
            other_context = other_text_embeddings

        latent, latents = self.init_latent(latent, batch_size)
        
        # set timesteps
        self.model.scheduler.set_timesteps(self.model_config["num_diffusion_steps"])
        all_latents = []

        object_mask = None
        self.diff_step = 0
        for i, t in enumerate(tqdm(self.model.scheduler.timesteps[-start_time:])):
            if uncond_embeddings_ is None:
                context = [uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings]
            else:
                context = [uncond_embeddings_, text_embeddings]
            if not self.model_config["low_resource"]:
                context = torch.cat(context)

            self.down_cross_index = 0
            self.mid_cross_index = 0
            self.up_cross_index = 0
            latents = self.diffusion_step(latents, context, t, other_context)

            if post_background and self.diff_step == self.args.background_blend_timestep:
                object_mask = Segmentor(self.controller,
                                        prompt,
                                        self.args.num_segments,
                                        self.args.background_segment_threshold,
                                        background_nouns=self.args.background_nouns)\
                    .get_background_mask(self.args.prompt.split(' ').index("{word}") + 1)
                self.enbale_attn_controller_changes = False
                mask = object_mask.astype(np.bool8) + orig_mask.astype(np.bool8)
                mask = torch.from_numpy(mask).float().cuda()
                shape = (1, 1, mask.shape[0], mask.shape[1])
                mask = torch.nn.Upsample(size=(64, 64), mode='nearest')(mask.view(shape))
                mask_eroded = dilate(mask.cpu().numpy()[0, 0], np.ones((3, 3), np.uint8), iterations=1)
                mask = torch.from_numpy(mask_eroded).float().cuda().view(1, 1, 64, 64)
                latents = mask * latents + (1 - mask) * orig_all_latents[self.diff_step]

            all_latents.append(latents)
            self.diff_step += 1

        if return_type == 'image':
            image = self.latent2image(latents)
        else:
            image = latents
        
        return image, latent, all_latents, object_mask
    
    
    def show_last_cross_attention(self, res: int, from_where: List[str], prompts, select: int = 0):
        show_cross_attention(self.controller, res, from_where, prompts, tokenizer=self.model.tokenizer, select=select)