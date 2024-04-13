# Kandinsky 2.2 ComfyUI Plugin

![](pics/workflow-simple.png)

Use the models of Kandinsky 2.2 published on 🤗 HuggingFace in ComfyUI.

## Features provided

Nodes provide options to combine prior and decoder models of Kandinsky 2.2.

- Find priors for text and images.
- Combine priors with weights.
- Prepare latents only or latents based on image (see img2img workflow).
- Use depth hint computed by a separate node.

All the weights can be found in Kandinsky Community on 🤗 HF in [Kandinsky 2.2 Collection](https://huggingface.co/collections/kandinsky-community/kandinsky-22-64f9d7de87c368f6184c73c9).

## Workflow Examples

- A simple text-based workflow ([source](workflows/workflow-simple.json)).

![](pics/workflow-simple.png)

- A workflow based on image prior embeds ([source](workflows/workflow-image-embed.json))

![](pics/workflow-image-embed.png)

- An Image-To-Image workflow ([source](workflows/workflow-img2img.json))

![](pics/workflow-img2img.png)

- A depth based workflow ([source](workflows/workflow-depth.json)) 

***Note: Don't forget to switch to kandinsky-2-2-controlnet-depth in decoder node.***

![](pics/workflow-depth.png)

## Installation

For the easiest experience, install the [Comfyui Manager](https://github.com/ltdrdata/ComfyUI-Manager) and 
use it to automate the installation process. The repository is not included in the list at the moment, but 
you'll need **Marigold depth estimation**, that can be installed via manager.

To install the plugin, open the terminal, cd to `<ComfyUI>/custom_nodes`, and clone the repo:
```
git clone https://github.com/vsevolod-oparin/comfyui-kandinsky22
```

Install the requirements using:
```
python -s -m pip install -r requirements.txt
```

### Download Models

Go to `models/checkpoints` in ComfyUI directory and run the command
```
git clone --depth 1 <HF repository>
```
E.g. to download needed checkpoints for the presented pipelines run the following
```
git clone --depth 1 https://huggingface.co/kandinsky-community/kandinsky-2-2-prior
git clone --depth 1 https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder
git clone --depth 1 https://huggingface.co/kandinsky-community/kandinsky-2-2-controlnet-depth
```

**Note 1**: Git won't show much of the progress. You'll need to wait till the models will be downloaded.

**Note 2**: Argument `--depth` can be skipped, but you're risking to download a lot of unnecessary data.


## Acknowledgments

**A special thanks to:**

- The developers of [Kandinsky Model](https://github.com/ai-forever/Kandinsky-2?tab=readme-ov-file#authors). 
- Patrick Von Platent for moving Kandinsky to [diffusers](https://github.com/huggingface/diffusers/issues/4290).  
- Comfyanonamous and the rest of the [ComfyUI](https://github.com/comfyanonymous/ComfyUI/tree/master) contributors for a fantastic UI!
