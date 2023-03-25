import os
from dataclasses import dataclass

import torch
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from src.attention_based_segmentation import Segmentor
from src.diffusion_model_wrapper import get_stable_diffusion_model, get_stable_diffusion_config, DiffusionModelWrapper
from src.null_text_inversion import invert_image
from src.prompt_to_prompt_controllers import AttentionStore

@dataclass
class SegmentationConfig:
    seed: int = 1111
    gpu_id: int = 0
    real_image_path: str = "real_images/rinon_cat.jpg"
    auth_token: str = ""
    low_resource: bool = True
    num_diffusion_steps: int = 50
    guidance_scale: float = 7.5
    max_num_words: int = 77
    prompt: str = "a cat in a basket"
    exp_path: str = "segmentation_results"

    num_segments: int = 5
    background_segment_threshold: float = 0.35

if __name__ == '__main__':
    args = SegmentationConfig()
    os.makedirs(args.exp_path, exist_ok=True)
    ldm_stable = get_stable_diffusion_model(args)
    ldm_stable_config = get_stable_diffusion_config(args)

    x_t = None
    uncond_embeddings = None
    if args.real_image_path != "":
        x_t, uncond_embeddings = invert_image(args, ldm_stable, ldm_stable_config, [args.prompt], args.exp_path)

    g_cpu = torch.Generator(device=ldm_stable.device).manual_seed(args.seed)
    controller = AttentionStore(ldm_stable_config["low_resource"])
    diffusion_model_wrapper = DiffusionModelWrapper(args, ldm_stable, ldm_stable_config, controller, generator=g_cpu)
    image, x_t, orig_all_latents, _ = diffusion_model_wrapper.forward([args.prompt],
                                                                      latent=x_t,
                                                                      uncond_embeddings=uncond_embeddings)
    segmentor = Segmentor(controller, [args.prompt], args.num_segments, args.background_segment_threshold)
    clusters = segmentor.cluster()
    cluster2noun = segmentor.cluster2noun(clusters)

    save_image(ToTensor()(image[0]), f"{args.exp_path}/image_rec.jpg")
    plt.imshow(clusters)
    plt.axis('off')
    plt.savefig(f"{args.exp_path}/segmentation.jpg", bbox_inches='tight', pad_inches=0)
