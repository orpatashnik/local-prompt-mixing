from __future__ import annotations

import gradio as gr
import nltk
import numpy as np
from PIL import Image

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from main import LPMConfig, main, setup

DESCRIPTION = '''# Localizing Object-level Shape Variations with Text-to-Image Diffusion Models
This is a demo for our ''Localizing Object-level Shape Variations with Text-to-Image Diffusion Models'' [paper](https://arxiv.org/abs/2303.11306).
We introduce a method that generates object-level shape variation for a given image. 
This demo supports both generated images and real images. To modify a real image, please upload it to the input image block and provide a prompt that describes its contents.

'''

stable, stable_config = setup(LPMConfig())

def main_pipeline(
        prompt: str,
        object_of_interest: str,
        proxy_words: str,
        number_of_variations: int,
        start_prompt_range: int,
        end_prompt_range: int,
        objects_to_preserve: str,
        background_nouns: str,
        seed: int,
        input_image: str):
        prompt = prompt.replace(object_of_interest, '{word}')
        proxy_words = proxy_words.split(',') if proxy_words != '' else []
        objects_to_preserve = objects_to_preserve.split(',') if objects_to_preserve != '' else []
        background_nouns = background_nouns.split(',') if background_nouns != '' else []
        args = LPMConfig(
            seed=seed,
            prompt=prompt,
            object_of_interest=object_of_interest,
            proxy_words=proxy_words,
            number_of_variations=number_of_variations,
            start_prompt_range=start_prompt_range,
            end_prompt_range=end_prompt_range,
            objects_to_preserve=objects_to_preserve,
            background_nouns=background_nouns,
            real_image_path="" if input_image is None else input_image
        )

        result_images, result_proxy_words = main(stable, stable_config, args)
        result_images = [im.permute(1, 2, 0).cpu().numpy() for im in result_images]
        result_images = [(im * 255).astype(np.uint8) for im in result_images]
        result_images = [Image.fromarray(im) for im in result_images]

        return result_images, ",".join(result_proxy_words)


with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)

    gr.HTML(
        '''<a href="https://huggingface.co/spaces/orpatashnik/local-prompt-mixing?duplicate=true">
        <img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>Duplicate the Space to run privately without waiting in queue''')

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Input image (optional)",
                type="filepath"
            )
            prompt = gr.Text(
                label='Prompt',
                max_lines=1,
                placeholder='A table below a lamp',
            )
            object_of_interest = gr.Text(
                label='Object of interest',
                max_lines=1,
                placeholder='lamp',
            )
            proxy_words = gr.Text(
                label='Proxy words - words used to obtain variations (a comma-separated list of words, can leave empty)',
                max_lines=1,
                placeholder=''
            )
            number_of_variations = gr.Slider(
                label='Number of variations (used only for automatic proxy-words)',
                minimum=2,
                maximum=30,
                value=7,
                step=1
            )
            start_prompt_range = gr.Slider(
                label='Number of steps before starting shape interval',
                minimum=0,
                maximum=50,
                value=7,
                step=1
            )
            end_prompt_range = gr.Slider(
                label='Number of steps before ending shape interval',
                minimum=1,
                maximum=50,
                value=17,
                step=1
            )
            objects_to_preserve = gr.Text(
                label='Words corresponding to objects to preserve (a comma-separated list of words, can leave empty)',
                max_lines=1,
                placeholder='table',
            )
            background_nouns = gr.Text(
                label='Words corresponding to objects that should be copied from original image (a comma-separated list of words, can leave empty)',
                max_lines=1,
                placeholder='',
            )
            seed = gr.Slider(
                label='Seed',
                minimum=1,
                maximum=100000,
                value=0,
                step=1
            )

            run_button = gr.Button('Generate')
        with gr.Column():
            result = gr.Gallery(label='Result').style(grid=4)
            proxy_words_result = gr.Text(label='Used proxy words')

            examples = [
                [
                    "hamster eating watermelon on the beach",
                    "watermelon",
                    "",
                    7,
                    6,
                    16,
                    "",
                    "hamster,beach",
                    48,
                    None
                ],
                [
                    "A decorated lamp in the livingroom",
                    "lamp",
                    "",
                    7,
                    4,
                    14,
                    "livingroom",
                    "",
                    42,
                    None
                ],
                [
                    "a snake in the field eats an apple",
                    "snake",
                    "",
                    7,
                    7,
                    17,
                    "apple",
                    "apple,field",
                    10,
                    None
                ]
            ]

            gr.Examples(examples=examples,
                        inputs=[
                            prompt,
                            object_of_interest,
                            proxy_words,
                            number_of_variations,
                            start_prompt_range,
                            end_prompt_range,
                            objects_to_preserve,
                            background_nouns,
                            seed,
                            input_image
                        ],
                        outputs=[
                            result,
                            proxy_words_result
                        ],
                        fn=main_pipeline,
                        cache_examples=False)


    inputs = [
        prompt,
        object_of_interest,
        proxy_words,
        number_of_variations,
        start_prompt_range,
        end_prompt_range,
        objects_to_preserve,
        background_nouns,
        seed,
        input_image
    ]
    outputs = [
        result,
        proxy_words_result
    ]
    run_button.click(fn=main_pipeline, inputs=inputs, outputs=outputs)

demo.queue(max_size=50).launch(share=False)