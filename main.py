import json
import os
from dataclasses import dataclass, field
from typing import List

import pyrallis
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from tqdm import tqdm

from src.diffusion_model_wrapper import DiffusionModelWrapper, get_stable_diffusion_model, get_stable_diffusion_config, \
    generate_original_image
from src.null_text_inversion import invert_image
from src.prompt_mixing import PromptMixing
from src.prompt_to_prompt_controllers import AttentionStore, AttentionReplace
from src.prompt_utils import get_proxy_prompts


def save_args_dict(args, similar_words):
    exp_path = os.path.join(args.exp_dir, args.prompt.replace(' ', '-'), f"seed={args.seed}_{args.exp_name}")
    os.makedirs(exp_path, exist_ok=True)

    args_dict = vars(args)
    args_dict['similar_words'] = similar_words
    with open(os.path.join(exp_path, "opt.json"), 'w') as fp:
        json.dump(args_dict, fp, sort_keys=True, indent=4)

    return exp_path

def setup(args):
    ldm_stable = get_stable_diffusion_model(args)
    ldm_stable_config = get_stable_diffusion_config(args)
    return ldm_stable, ldm_stable_config


def main(ldm_stable, ldm_stable_config, args):

    similar_words, prompts, another_prompts = get_proxy_prompts(args, ldm_stable)
    exp_path = save_args_dict(args, similar_words)

    images = []
    x_t = None
    uncond_embeddings = None

    if args.real_image_path != "":
        ldm_stable, ldm_stable_config = setup(args)
        x_t, uncond_embeddings = invert_image(args, ldm_stable, ldm_stable_config, prompts, exp_path)

    image, x_t, orig_all_latents, orig_mask, average_attention = generate_original_image(args, ldm_stable, ldm_stable_config, prompts, x_t, uncond_embeddings)
    save_image(ToTensor()(image[0]), f"{exp_path}/{similar_words[0]}.jpg")
    save_image(torch.from_numpy(orig_mask).float(), f"{exp_path}/{similar_words[0]}_mask.jpg")
    images.append(image[0])

    object_of_interest_index = args.prompt.split().index('{word}') + 1
    pm = PromptMixing(args, object_of_interest_index, average_attention)

    do_other_obj_self_attn_masking = len(args.objects_to_preserve) > 0 and args.end_preserved_obj_self_attn_masking > 0
    do_self_or_cross_attn_inject = args.cross_attn_inject_steps != 0.0 or args.self_attn_inject_steps != 0.0
    if do_other_obj_self_attn_masking:
        print("Do self attn other obj masking")
    if do_self_or_cross_attn_inject:
        print(f'Do self attn inject for {args.self_attn_inject_steps} steps')
        print(f'Do cross attn inject for {args.cross_attn_inject_steps} steps')

    another_prompts_dataloader = DataLoader(another_prompts[1:], batch_size=args.batch_size, shuffle=False)

    for another_prompt_batch in tqdm(another_prompts_dataloader):
        batch_size = len(another_prompt_batch["word"])
        batch_prompts = prompts * batch_size
        batch_another_prompt = another_prompt_batch["prompt"]
        if do_self_or_cross_attn_inject or do_other_obj_self_attn_masking:
            batch_prompts.append(prompts[0])
            batch_another_prompt.insert(0, prompts[0])

        if do_self_or_cross_attn_inject:
            controller = AttentionReplace(batch_another_prompt, ldm_stable.tokenizer, ldm_stable.device,
                                          ldm_stable_config["low_resource"], ldm_stable_config["num_diffusion_steps"],
                                          cross_replace_steps=args.cross_attn_inject_steps,
                                          self_replace_steps=args.self_attn_inject_steps)
        else:
            controller = AttentionStore(ldm_stable_config["low_resource"])

        diffusion_model_wrapper = DiffusionModelWrapper(args, ldm_stable, ldm_stable_config, controller, prompt_mixing=pm)
        with torch.no_grad():
            image, x_t, _, mask = diffusion_model_wrapper.forward(batch_prompts, latent=x_t, other_prompt=batch_another_prompt,
                                                                  post_background=args.background_post_process, orig_all_latents=orig_all_latents,
                                                                  orig_mask=orig_mask, uncond_embeddings=uncond_embeddings)

        for i in range(batch_size):
            image_index = i + 1 if do_self_or_cross_attn_inject or do_other_obj_self_attn_masking else i
            save_image(ToTensor()(image[image_index]), f"{exp_path}/{another_prompt_batch['word'][i]}.jpg")
            if mask is not None:
                save_image(torch.from_numpy(mask).float(), f"{exp_path}/{another_prompt_batch['word'][i]}_mask.jpg")
            images.append(image[image_index])

    images = [ToTensor()(image) for image in images]
    save_image(images, f"{exp_path}/grid.jpg", nrow=min(max([i for i in range(2, 8) if len(images) % i == 0]), 8))
    return images, similar_words


@dataclass
class LPMConfig:

    # general config
    seed: int = 10
    batch_size: int = 1
    exp_dir: str = "results"
    exp_name: str = ""
    display_images: bool = False
    gpu_id: int = 0

    # Stable Diffusion config
    auth_token: str = ""
    low_resource: bool = True
    num_diffusion_steps: int = 50
    guidance_scale: float = 7.5
    max_num_words: int = 77

    # prompt-mixing
    prompt: str = "a {word} in the field eats an apple"
    object_of_interest: str = "snake"                                   # The object for which we generate variations
    proxy_words: List[str] = field(default_factory=lambda :[])          # Leave empty for automatic proxy words
    number_of_variations: int = 20
    start_prompt_range: int = 7                                         # Number of steps to begin prompt-mixing
    end_prompt_range: int = 17                                          # Number of steps to finish prompt-mixing

    # attention based shape localization
    objects_to_preserve: List[str] = field(default_factory=lambda :[])  # Objects for which apply attention based shape localization
    remove_obj_from_self_mask: bool = True                              # If set to True, removes the object of interest from the self-attention mask
    obj_pixels_injection_threshold: float = 0.05
    end_preserved_obj_self_attn_masking: int = 40

    # real image
    real_image_path: str = ""

    # controllable background preservation
    background_post_process: bool = True
    background_nouns: List[str] = field(default_factory=lambda :[])     # Objects to take from the original image in addition to the background
    num_segments: int = 5                                               # Number of clusters for the segmentation
    background_segment_threshold: float = 0.3                           # Threshold for the segments labeling
    background_blend_timestep: int = 35                                 # Number of steps before background blending

    # other
    cross_attn_inject_steps: float = 0.0
    self_attn_inject_steps: float = 0.0


if __name__ == '__main__':
    args = pyrallis.parse(config_class=LPMConfig)

    print(args)

    stable, stable_config = setup(args)
    main(stable, stable_config, args)
