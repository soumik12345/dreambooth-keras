# Implementation of DreamBooth using KerasCV and TensorFlow

> This repository is a fork of [sayakpaul/dreambooth-keras](https://github.com/sayakpaul/dreambooth-keras) developed by [Sayak Paul](https://github.com/sayakpaul) and [Chansung Park](https://github.com/deep-diver). Please star ⭐️ the original repository.

This repository provides an implementation of [DreamBooth](https://arxiv.org/abs/2208.12242) using KerasCV and TensorFlow. The implementation is heavily referred from Hugging Face's `diffusers` [example](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth).

DreamBooth is a way of quickly teaching (fine-tuning) Stable Diffusion about new visual concepts. For more details, refer to [this document](https://dreambooth.github.io/).

**The code provided in this repository is for research purposes only**. Please check out [this section](https://github.com/keras-team/keras-cv/tree/master/keras_cv/models/stable_diffusion#uses) to know more about the potential use cases and limitations.

By loading this model you accept the CreativeML Open RAIL-M license at https://raw.githubusercontent.com/CompVis/stable-diffusion/main/LICENSE.

<div align="center">
<img src="https://i.imgur.com/gYlgLPm.png"/>
</div>

If you're just looking for the accompanying resources of this repository, here are the links:

* [Inference Colab Notebook](https://colab.research.google.com/github/sayakpaul/dreambooth-keras/blob/main/notebooks/inference_dreambooth.ipynb)
* [Blog post on keras.io](https://keras.io/examples/generative/dreambooth/)
* [Fine-tuned model weights](https://huggingface.co/chansung/dreambooth-dog)

### Table of contents

* [Performing DreamBooth training with the codebase](#steps-to-perform-dreambooth-training-using-the-codebase)
* [Running inference](#inference)
* [Results](#results)
* [Using in Diffusers 🧨](#using-in-diffusers-)
* [Notes](#notes-on-preparing-data-for-dreambooth-training-of-faces)
* [Acknowledgements](#acknowledgements)

**Update 15/02/2023**: Thanks to [Soumik Rakshit](https://in.linkedin.com/in/soumikrakshit); we now have better utilities to support Weights and Biases (see https://github.com/sayakpaul/dreambooth-keras/pull/22).

## Steps to perform DreamBooth training using the codebase

1. You can install the library using `pip install git+https://github.com/soumik12345/dreambooth-keras.git`

2. You first need to choose a class to which a unique identifier is appended. This repository codebase was tested using `sks` as the unique idenitifer and `dog` as the class.

    Then two types of prompts are generated: 

    (a) **instance prompt**: f"a photo of {self.unique_id} {self.class_category}"
    (b) **class prompt**: f"a photo of {self.class_category}"

3. **Instance images**
    
    Get a few images (3 - 10) that are representative of the concept the model is going to be fine-tuned with. These images would be associated with the `instance_prompt`. These images are referred to as the `instance_images` from the codebase. Archive these images and host them somewhere online such that the archive can be downloaded using `tf.keras.utils.get_file()` function internally.

4. **Class images**
    
    DreamBooth uses prior-preservation loss to regularize training. Long story cut short,
prior-preservation loss helps the model to slowly adapt to the new concept under consideration from any prior knowledge it may have had about the concept. To use prior-preservation loss, we need the class prompt as shown above. The class prompt is used to generate a pre-defined number of images which are used for computing the final loss used for DreamBooth training. 

    As per [this resource](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth), 200 - 300 images generated using the class prompt work well for most cases. 

    So, after you have decided `instance_prompt` and `class_prompt`, use [this Colab Notebook](https://colab.research.google.com/github/sayakpaul/dreambooth-keras/blob/main/notebooks/generate_class_priors.ipynb) to generate some images that would be used for training with the prior-preservation loss. Then archive the generated images as a single archive and host it online such that it can be downloaded using using `tf.keras.utils.get_file()` function internally. In the codebase, we simply refer to these images as `class_images`.
    
> It's possible to conduct DreamBooth training WITHOUT using a prior preservation loss. This repository always uses it. For people to easily test this codebase, we hosted the instance and class images [here](https://huggingface.co/datasets/sayakpaul/sample-datasets/tree/main). 

5. Launch training! There are a number of hyperparameters you can play around with. Refer to the `train_dreambooth.py` script to know more about them. Here's a command that launches training with mixed-precision and other default values:

    ```bash
    python train_dreambooth.py --mp
    ```

    You can also fine-tune the text encoder by specifying the `--train_text_encoder` option. 

    Additionally, the script supports integration with [Weights and Biases (`wandb`)](https://wandb.ai/). If you specify `--log_wandb`,
    - it will automatically log the training metrics to your `wandb` dashboard using the [`WandbMetricsLogger` callback](https://docs.wandb.ai/guides/integrations/keras#experiment-tracking-with-wandbmetricslogger).
    - it will also upload your model checkpoints at the end of each epoch to your `wandb` project as an [artifacts](https://docs.wandb.ai/guides/artifacts) for model versioning. This is done using the `DreamBoothCheckpointCallback` which was built using [`WandbModelCheckpoint` callback](https://docs.wandb.ai/guides/integrations/keras#model-checkpointing-using-wandbmodelcheckpoint).
    - it will also perform inference with the DreamBoothed model parameters at the end of each epoch and log them into a [`wandb.Table`](https://docs.wandb.ai/guides/data-vis) in your `wandb` dashboard. This is done using the `QualitativeValidationCallback`, which also logs generated images into a media panel on your `wandb` dashboard at the end of the training.

    Here's a command that launches training and logs training metrics and generated images to your Weights & Biases workspace:

    ```bash
    python train_dreambooth.py \
      --log_wandb \
      --validation_prompts \
        "a photo of sks dog with a cat" \
        "a photo of sks dog riding a bicycle" \
        "a photo of sks dog peeing" \
        "a photo of sks dog playing cricket" \
        "a photo of sks dog as an astronaut"
    ```

    Additionally, you can also have you datasets corresponding to instance and class images stored as [artifacts](https://docs.wandb.ai/guides/artifacts) for versioning your dataset and tracking the lineage of your workflow. You can specify the artifact addresses of your datasets in the corresponding flags, like the following example:

    ```bash
    python train_dreambooth.py \
      --instance_images_url "geekyrakshit/dreambooth-keras/monkey-instance-images:v0" \
      --class_images_url "geekyrakshit/dreambooth-keras/monkey-class-images:v0" \
      --class_category "monkey" \
      --mp \
      --log_wandb \
      --lr 5e-06 \
      --max_train_steps 2000 \
      --validation_prompts \
          "a photo of sks monkey with a cat" \
          "a photo of sks monkey riding a bicycle" \
          "a photo of sks monkey as an astronaut" \
          "a photo of sks monkey in front of the taj mahal" \
          "a photo of sks monkey wearing sunglasses and drinking beer"
    ```

    [Here's](https://wandb.ai/geekyrakshit/dreambooth-keras/runs/huou7nzr) an example `wandb` run where you can find the generated images as well as the [model checkpoints](https://wandb.ai/geekyrakshit/dreambooth-keras/artifacts/model/run_huou7nzr_model).

## Inference

* [Colab Notebook](https://colab.research.google.com/github/sayakpaul/dreambooth-keras/blob/main/notebooks/inference_dreambooth.ipynb)
* [Script for launching bulk experiments](https://github.com/sayakpaul/dreambooth-keras/blob/main/scripts/generate_experimental_images.py)

## Results

We have tested our implementation in two different methods: (a) fine-tuning the diffusion model (the UNet) only, (b) fine-tuning the diffusion model along with the text encoder. The experiments were conducted over a wide range of hyperparameters for `learning rate` and `training steps` for during training and for `number of steps` and `unconditional guidance scale` (ugs) during inference. But only the most salient results (from our perspective) are included here. If you are curious about how different hyperparameters affect the generated image quality, find the link to the full reports in each section.

__Note that our experiments were guided by [this blog post from Hugging Face](https://huggingface.co/blog/dreambooth).__

### (a) Fine-tuning diffusion model

Here are a selected few results from various experiments we conducted. Our experimental logs for this setting are available [here](https://wandb.ai/sayakpaul/dreambooth-keras). More visualization images (generated with the checkpoints from these experiments) are available [here](https://wandb.ai/sayakpaul/experimentation_images). 


<div align="center">
<table>
  <tr>
    <th>Images</th>
    <th>Steps</th>
    <th>UGS</th>
    <th>Setting</th>
  </tr>
  <tr>
    <td><img src="https://i.imgur.com/UUSfrwW.png"/></td>
    <td>50</td>
    <td>30</td>
    <td>LR: 1e-6 Training steps: 800 <a href="https://huggingface.co/sayakpaul/dreambooth-keras-dogs-unet/resolve/main/lr_1e-6_steps_800_unet.h5">(Weights)</a></td>
  </tr>
  <tr>
    <td><img src="https://i.imgur.com/Ewt0BhG.png"/></td>
    <td>25</td>
    <td>15</td>
    <td>LR: 1e-6 Training steps: 1000 <a href="https://huggingface.co/sayakpaul/dreambooth-keras-dogs-unet/resolve/main/lr_1e-6_steps_1000.h5">(Weights)</a></td>
  </tr>  
  <tr>
    <td><img src="https://i.imgur.com/Dn0uGZa.png"/></td>
    <td>75</td>
    <td>15</td>
    <td>LR: 3e-6 Training steps: 1200 <a href="https://huggingface.co/sayakpaul/dreambooth-keras-dogs-unet/resolve/main/lr_3e-6_steps_1200_unet.h5">(Weights)</a></td>
  </tr>
</table>
<sub><b>Caption</b>: "A photo of sks dog in a bucket" </sub> 
</div>

### (b) Fine-tuning text encoder + diffusion model

<div align="center">
<table>
  <tr>
    <th>Images</th>
    <th>Steps</th>
    <th>ugs</th>
  </tr>
  <tr>
    <td><img src="https://i.ibb.co/BNVtwDB/dog.png"/></td>
    <td>75</td>
    <td>15</td>
  </tr>
  <tr>
    <td><img src="https://i.ibb.co/zWMzxq2/dog-2.png"/></td>
    <td>75</td>
    <td>30</td>
  </tr>  
</table>
<sub>"<b>Caption</b>: A photo of sks dog in a bucket" </sub> 

<sub> w/ learning rate=9e-06, max train steps=200 (<a href="https://huggingface.co/chansung/dreambooth-dog">weights</a> | <a href="https://wandb.ai/chansung18/dreambooth-keras-generating-images?workspace=user-chansung18">reports</a>)</sub>
</div><br>


<div align="center">
<table>
  <tr>
    <th>Images</th>
    <th>Steps</th>
    <th>ugs</th>
  </tr>
  <tr>
    <td><img src="https://i.ibb.co/XYz3s5N/chansung.png"/></td>
    <td>150</td>
    <td>15</td>
  </tr>
  <tr>
    <td><img src="https://i.ibb.co/mFMZG04/chansung-2.png"/></td>
    <td>75</td>
    <td>30</td>
  </tr>  
</table>
<sub>"<b>Caption</b>: A photo of sks person without mustache, handsome, ultra realistic, 4k, 8k"</sub> 

<sub> w/ learning rate=9e-06, max train steps=200 (<a href="https://huggingface.co/datasets/chansung/me">datasets</a> | <a href="https://wandb.ai/chansung18/dreambooth-generate-me?workspace=user-chansung18">reports</a>)</sub>
</div><br>

## Using in Diffusers 🧨

The [`diffusers` library](https://github.com/huggingface/diffusers/) provides state-of-the-art tooling for experimenting with
different Diffusion models, including Stable Diffusion. It includes 
different optimization techniques that can be leveraged to perform efficient inference
with `diffusers` when using large Stable Diffusion checkpoints. One particularly 
advantageous feature `diffusers` has is its support for [different schedulers](https://huggingface.co/docs/diffusers/using-diffusers/schedulers) that can
be configured during runtime and can be integrated into any compatible Diffusion model.

Once you have obtained the DreamBooth fine-tuned checkpoints using this codebase, you can actually
export those into a handy [`StableDiffusionPipeline`](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/overview) and use it from the `diffusers` library directly. 

Consider this repository: [chansung/dreambooth-dog](https://huggingface.co/chansung/dreambooth-dog). You can use the
checkpoints of this repository in a `StableDiffusionPipeline` after running some small steps:

```py
from diffusers import StableDiffusionPipeline

# checkpoint of the converted Stable Diffusion from KerasCV
model_ckpt = "sayakpaul/text-unet-dogs-kerascv_sd_diffusers_pipeline"
pipeline = StableDiffusionPipeline.from_pretrained(model_ckpt)
pipeline.to("cuda")

unique_id = "sks"
class_label = "dog"
prompt = f"A photo of {unique_id} {class_label} in a bucket"
image = pipeline(prompt, num_inference_steps=50).images[0]
```

Follow [this guide](https://huggingface.co/docs/diffusers/main/en/using-diffusers/kerascv) to know more. 


### Experimental results through various scheduler settings:

We have converted fine-tuned checkpoint for the dog images into Diffusers compatible StableDiffusionPipeline and ran various experiments with different scheduler settings. For example, the following parameters of the `DDIMScheduler` are tested on a different set of `guidance_scale` and `num_inference_steps`.

```python
num_inference_steps_list = [25, 50, 75, 100]
guidance_scale_list = [7.5, 15, 30]

scheduler_configs = {
  "DDIMScheduler": {
      "beta_value": [
          [0.000001, 0.02], 
          [0.000005, 0.02], 
          [0.00001, 0.02], 
          [0.00005, 0.02], 
          [0.0001, 0.02], 
          [0.0005, 0.02]
      ],
      "beta_schedule": [
          "linear", 
          "scaled_linear", 
          "squaredcos_cap_v2"
      ],
      "clip_sample": [True, False],
      "set_alpha_to_one": [True, False],
      "prediction_type": [
          "epsilon", 
          "sample", 
          "v_prediction"
      ]
  }
}
```

Below is the comparison between different values of `beta_schedule` parameters while others are fixed to their default values. Take a look at [the original report](https://docs.google.com/spreadsheets/d/1_NhWuORn5ByEnvD9T3X4sHUnz_GR8uEtbE5HbI98hOM/edit?usp=sharing) which includes the results from other schedulers such as `PNDMScheduler` and `LMSDiscreteScheduler`. 

It is often observed the default settings do guarantee to generate better quality images. For example, the default values of `guidance_scale` and `beta_schedule` are set to 7.5 and `linear`. However, when `guidance_scale` is set to 7.5, `scaled_linear` of the `beta_schedule` seems to work better. Or, when `beta_schedule` is set to `linear`, higher `guidance_scale` seems to work better. 

![](https://i.postimg.cc/QsW-CKTcv/DDIMScheduler.png)

We ran 4,800 experiments which generated 38,400 images in total. Those experiments are logged in Weights and Biases. If you are curious, do check them out [here](https://wandb.ai/chansung18/SD-Scheduler-Explore?workspace=user-chansung18) as well as the [script](https://gist.github.com/deep-diver/0a2deb2cd369ab8c1bf3ee12f47d272a) that was used to run the experiments. 

## Notes on preparing data for DreamBooth training of faces

In addition to the tips and tricks shared in [this blog post](https://huggingface.co/blog/dreambooth#using-prior-preservation-when-training-faces), we followed these things while preparing the instances for conducting DreamBooth training on human faces:

* Instead of 3 - 5 images, use 20 - 25 images of the same person varying different angles, backgrounds, and poses.
* No use of images containing multiple persons. 
* If the person wears glasses, don't include images only with glasses. Combine images with and without glasses.

Thanks to [Abhishek Thakur](https://no.linkedin.com/in/abhi1thakur) for sharing these tips. 

## Acknowledgements

* Thanks to Hugging Face for providing the [original example](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth). It's very readable and easy to understand.
* Thanks to the ML Developer Programs' team at Google for providing GCP credits.
