import os
from glob import glob

import keras_cv
from typing import List

import PIL

import tensorflow as tf

import wandb
from wandb.keras import WandbModelCheckpoint


def fetch_wandb_artifact(artifact_address: str, artifact_type: str):
    return (
        wandb.Api().artifact(artifact_address, type=artifact_type).download()
        if wandb.run is None
        else wandb.use_artifact(artifact_address, type=artifact_type).download()
    )


def load_model_from_wandb_artifact(artifact_address: str, image_resolution: int):
    model_artifact_dir = fetch_wandb_artifact(
        artifact_address=artifact_address, artifact_type="model"
    )
    sd_model = keras_cv.models.StableDiffusion(
        img_height=image_resolution, img_width=image_resolution
    )
    unet_checkpoint_files = glob(os.path.join(model_artifact_dir, "*-unet.h5"))
    text_encoder_checkpoint_files = glob(
        os.path.join(model_artifact_dir, "*-text_encoder.h5")
    )
    if len(unet_checkpoint_files) > 0:
        sd_model.diffusion_model.load_weights(unet_checkpoint_files[0])
    if len(text_encoder_checkpoint_files) > 0:
        sd_model.text_encoder.load_weights(text_encoder_checkpoint_files[0])
    return sd_model


class QualitativeValidationCallback(tf.keras.callbacks.Callback):
    def __init__(
        self,
        img_heigth: int,
        img_width: int,
        prompts: List[str],
        num_imgs_to_gen: int = 5,
        num_diffusion_steps: int = 50,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.img_heigth = img_heigth
        self.img_width = img_width
        self.prompts = prompts
        self.num_imgs_to_gen = num_imgs_to_gen
        self.num_diffusion_steps = num_diffusion_steps
        self.sd_model = keras_cv.models.StableDiffusion(
            img_height=self.img_heigth, img_width=self.img_width
        )
        self.wandb_table = wandb.Table(columns=["epoch", "prompt", "images"])

    def on_epoch_end(self, epoch, logs=None):
        print(f"Performing inference for logging generated images for epoch {epoch}...")
        print(f"Number of images to generate: {self.num_imgs_to_gen}")

        # load weights to stable diffusion model
        self.sd_model.diffusion_model.set_weights(
            self.model.diffusion_model.get_weights()
        )
        if hasattr(self.model, "text_encoder"):
            self.sd_model.text_encoder.set_weights(
                self.model.text_encoder.get_weights()
            )

        for prompt in self.prompts:
            images_dreamboothed = self.sd_model.text_to_image(
                prompt,
                batch_size=self.num_imgs_to_gen,
                num_steps=self.num_diffusion_steps,
            )
            images_dreamboothed = [
                wandb.Image(PIL.Image.fromarray(image), caption=f"{i}: {prompt}")
                for i, image in enumerate(images_dreamboothed)
            ]
            self.wandb_table.add_data(epoch, prompt, images_dreamboothed)
            wandb.log({f"validation/Prompt: {prompt}": images_dreamboothed})

    def on_train_end(self, logs=None):
        wandb.log({"validation-table": self.wandb_table})


class DreamBoothCheckpointCallback(WandbModelCheckpoint):
    def __init__(
        self, filepath, save_weights_only: bool = False, *args, **kwargs
    ) -> None:
        super(DreamBoothCheckpointCallback.__bases__[0], self).__init__(
            filepath, save_weights_only=save_weights_only, *args, **kwargs
        )
        self.save_weights_only = save_weights_only
        # User-friendly warning when trying to save the best model.
        if self.save_best_only:
            self._check_filepath()
        self._is_old_tf_keras_version = None

    def _log_ckpt_as_artifact(self, filepath: str, aliases) -> None:
        if wandb.run is not None:
            model_artifact = wandb.Artifact(f"run_{wandb.run.id}_model", type="model")
            for file in glob(f"{filepath}*.h5"):
                model_artifact.add_file(file)
            wandb.log_artifact(model_artifact, aliases=aliases or [])
