import os
import PIL
import shutil
import hashlib
import argparse
import numpy as np
from tqdm import tqdm

import wandb
import keras_cv
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to generate class priors for DreamBooth training using Stable Diffusion."
    )
    parser.add_argument("--image_resolution", default=512, type=int)
    parser.add_argument("--class_prompt", default="a photo of monkey", type=str)
    parser.add_argument("--num_imgs_to_generate", default=300, type=int)
    parser.add_argument("--batch_size", default=3, type=int)
    parser.add_argument("--wandb_artifact_name", default="monkey-instance", type=str)

    return parser.parse_args()


def run(args):
    wandb.init(project="dreambooth-keras", job_type="inference")

    config = wandb.config
    config.image_resolution = args.image_resolution
    config.class_prompt = args.class_prompt
    config.num_imgs_to_generate = args.num_imgs_to_generate
    config.batch_size = args.batch_size
    config.wandb_artifact_name = args.wandb_artifact_name

    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    model = keras_cv.models.StableDiffusion(
        img_width=config.image_resolution,
        img_height=config.image_resolution,
        jit_compile=True,
    )

    os.makedirs("class-images", exist_ok=True)

    for i in tqdm(range(config.num_imgs_to_generate)):
        images = model.text_to_image(
            config.class_prompt,
            batch_size=config.batch_size,
        )
        idx = np.random.choice(len(images))
        selected_image = PIL.Image.fromarray(images[idx])

        hash_image = hashlib.sha1(selected_image.tobytes()).hexdigest()
        image_filename = os.path.join("class-images", f"{hash_image}.jpg")
        selected_image.save(image_filename)

    artifact = wandb.Artifact(config.wandb_artifact_name, type="dataset")
    artifact.add_dir("class-images")
    wandb.log_artifact(artifact)

    wandb.finish()

    shutil.rmtree("class-images")


if __name__ == "__main__":
    args = parse_args()
    run(args)
