# ComfyUI-TacoNodes

<img alt="Example Gif" src="https://raw.githubusercontent.com/YOUR-WORST-TACO/ComfyUI-TacoNodes/assets/TIAL_adv.gif" width="512" />

A handful of ComfyUI nodes I created for my own purposes.

## Installation
Navigate to your ComfyUI installation location and find the `custom_nodes` folder.

While inside the `custom_nodes` folder run the following command:
```
git clone https://github.com/YOUR-WORST-TACO/ComfyUI-TacoNodes.git
```

Restart ComfyUI to see the new nodes, located under `Taco_Nodes`

## Future Goals
- [ ] Allow **Taco Animated Image Loader** to provide a mask output
- [ ] Create a method to inherit latent data from previous latent
- [ ] Possibly refactor how the **Taco Latent Image** options are formatted
- [ ] Add any other node I find I need

## Usage

- [Taco Latent Image](#taco-latent-image)
  - `Taco_Nodes/Taco Latent Image`
  - A replacement for the standard **Empty Latent Image** intended for faster latent image creation.
- [Taco Animated Image Loader](#taco-animated-image-loader)
  - `Taco_Nodes/Gifs/Taco Animated Image Loader`
  - Loads an animated image and batches each frame for processing.
- [Taco Img2Img Animated Loader](#taco-img2img-animated-loader)
  - `Taco_Nodes/Gifs/Taco Img2Img Animated Loader`
  - Loads a standard image and creates a batch for processing.
- [Taco_Img2Img Animated Processor](#taco-img2img-animated-processor)
  - `Taco_Nodes/Gifs/Taco Img2Img Animated Processor`
  - Takes an input image Tensor and automatically batches it for processing.
- [Taco Gif Maker](#taco-gif-maker)
  - `Taco_Nodes/Gifs/Taco Gif Maker`
  - Takes a collection of images and combines them into a gif.

## Taco Latent Image

I made this to simplify my tweaking of workflows in ComfyUI as I found having to remember exact pixel measurements 
frustrating and inconvenient

To use this you first select an aspect ratio, then select a scale for that ratio which will multiply the aspect ratio

### Aspect Ratios
This is a list of the following Aspect Ratios and their corresponding resolutions

| Aspect Ratio | Resolution |
|--------------|------------|
| 1:1          | 512 x 512  |
| 3:4          | 416 x 576  |
| 4:3          | 576 x 416  |
| 9:16         | 384 x 672  |
| 16:9         | 672 x 384  |

### Resolution Ratio
We take the aspect ratio, ie `1:1` and get its resolution `512px x 512px` then multiply the resolution by the Ratio, 
ie `2` which results in the final resolution of `1024px x 1024px`.

| Resolution   | Ratio | Use-case                                                          |
|--------------|-------|-------------------------------------------------------------------|
| small (512)  | 1     | great for small latents and initial images for hiresfix workflows |
| medium (1024 | 2     | good for standard renders and works well for SDXL 1.0             |
| large (2048) | 4     | I don't recommend using this unless you know what you are doing   |

## Taco Animated Image Loader

Allows you to input an animated image `(must be supported by PIL)` and breaks it up into a batch of the individual 
frames.

### Example Workflow

<img alt="TAIL Workflow" src="https://raw.githubusercontent.com/YOUR-WORST-TACO/ComfyUI-TacoNodes/assets/TAIL_workflow.png" />

*Enjoy my hideous creation*

<img alt="TAIL Gif" src="https://raw.githubusercontent.com/YOUR-WORST-TACO/ComfyUI-TacoNodes/assets/TAIL.gif" width="512" />

## Taco Img2Img Animated Loader

Allows you to take a single image and generate a batch of images to be processed for gif creation, I like to use this 
to spice up a still.

### Example Workflow

<img alt="TIAL Workflow" src="https://raw.githubusercontent.com/YOUR-WORST-TACO/ComfyUI-TacoNodes/assets/TIAL_workflow.png" />

<img alt="TIAL Gif" src="https://raw.githubusercontent.com/YOUR-WORST-TACO/ComfyUI-TacoNodes/assets/TIAL.gif" width="512" />

## Taco Img2Img Animated Processor

Allows you to pass in an existing image Tensor and batch it by frames for further processing.

### Example Workflow

I first generate an image with KSampler, then decode that image and feed it into the processor, then process that batch.

<img alt="TIAP Workflow" src="https://raw.githubusercontent.com/YOUR-WORST-TACO/ComfyUI-TacoNodes/assets/TIAP_workflow.png" />

<img alt="TIAP Gif" src="https://raw.githubusercontent.com/YOUR-WORST-TACO/ComfyUI-TacoNodes/assets/TIAP.gif" width="512" />

## Taco GIf Maker

Takes an input of **IMAGE** and will combine the input into a gif at the desired frame rate, see the above examples for 
Gif Maker usage.


## Advanced Examples

### Taco Img2Img Animated Loader with masking

<img alt="TIAL Workflow" src="https://raw.githubusercontent.com/YOUR-WORST-TACO/ComfyUI-TacoNodes/assets/TIAL_adv_workflow.png" />

<img alt="TIAL Gif" src="https://raw.githubusercontent.com/YOUR-WORST-TACO/ComfyUI-TacoNodes/assets/TIAL_adv.gif" width="512" />

## Liscense
Copyright [2023] [YOUR-WORST-TACO]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
