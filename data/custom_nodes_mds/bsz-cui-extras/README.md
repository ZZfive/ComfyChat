# bsz-cui-extras
Addons for [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## Custom Nodes
To install all the nodes, simply put the entire `bsz-nodes` folder inside your `custom_nodes` folder in ComfyUI

To install specific nodes, you may put individual `.py` files from `bsz-nodes` directly into the ComfyUI `custom_nodes` folder.

`__init__.py` simply forwards all nodes within its folder to ComfyUI, and is not necessary if you're putting nodes directly into `custom_nodes`

### bsz-auto-hires.py
Contains 3 nodes each with a different means to the same end result.
These nodes are designed to automatically calculate the appropriate latent sizes when performing a "Hi Res Fix" style workflow.

#### Common:
  - Input
    - `base_model_res` : Resolution of base model being used. SD 1.5 ≅ 512, SD 2.1 ≅ 768, SDXL ≅ 1024
  - Output
    - `Lo Res Width` : Width intended to be used for first/low res pass
    - `Lo Res Height` : Height intended to be used for first/low res pass
    - `Hi Res Width` : Width intended to be used for final/high res pass
    - `Hi Res Height` : Height intended to be used for final/high res pass

#### BSZAbsoluteHires:
  - Input
    - `desired_width` : Width in pixels for final/high res pass.
    - `desired_height` : Height in pixels for final/high res pass.

#### BSZAspectHires:
  - Input
    - `desired_aspect_x` : Horizontal aspect.
    - `desired_aspect_Y` : Vertical aspect.
    - `scale` : Hi Res horizontal and vertical scale over Lo Res sizes. Note that because this scales both axes, a scale of `2.0` will actually quadruple the amount of pixels in an image, so use with care.

#### BSZCombinedHires:
A unique node that functions both as BSZAbsoluteHires and BSZAspectHires with a convenient toggle
  - Input
    - `use_aspect_scale` : Use aspect & scale inputs instead of desired width/height inputs

### bsz-principled.py
All-in-one nodes for sampling and scaling pipelines.

#### BSZPrincipledSampler:
Text2Image, Image2Image pipeline workflows with optional refiner
Input fields
  - `base_model` : Model from base checkpoint
  - `base_clip` : CLIP from base checkpoint
  - `latent` : Latent image to start from
  - `refiner_model` : Model from refiner checkpoint. **Optional**
  - `refiner_clip` : CLIP from refiner checkpoint. **Optional**
  - `positive_prompt` : Positive prompt
  - `negative_prompt` : Negative prompt
  - `steps` : Steps for non-scaled pass
  - `denoise` : Denoise amount for latent input. Recommend 0.3 for img2img/pixel scale, 0.6 for latent scale
  - `cfg` : CFG scale
  - `refiner_amount` : Refiner to base ratio. Requires refiner model and refiner clip to function
  - `refiner_ascore_positive` : Refiner aesthetic score for positive prompt. Ignored if refiner is not XL architecture
  - `refiner_ascore_negative` : Refiner aesthetic score for negative prompt. Ignored if refiner is not XL architecture
  - `sampler` : Sampler. DDIM or Euler needed for proper refiner usage
  - `scheduler` : Scheduler. Normal needed for proper refiner usage
  - `seed` : Seedy.
Additionally, it produces batches by seed increment instead of whatever the hell ComfyUI does by default. This means seed 4 batch index 3 is equivalent to seed 7, making it much easier to reproduce images from batches.

#### BSZPrincipledScale:
Up/downscaling with either pixel, latent, or model methods. Pixel and model methods first decode with the VAE before scaling and re-encoding.
 - `vae` : VAE to use when converting to pixel space and back
 - `latent` : Latnet image
 - `width` : New width
 - `height` : New height
 - `method` : Scaling method to use

### bsz-latent-manipulation.py
Nodes for manipulating the color of latent images.

#### BSZColoredLatentImageXL
Creates an colored (non-empty) latent image according to the SDXL VAE
  - Input
    - `color` : Choice of color.
    - `strength` : Color strength/opacity over zero/gray
    - `width/height/batch_size`: Same as EmptyLatentImage

#### BSZLatentOffsetXL
Offsets the latent image(s) value towards black/white according to the SDXL VAE
  - Input
    - `latent` : Latent image(s).
    - `offset` : 0.0 is unchanged, -1.0 is black, 1.0 is white.

#### BSZLatentRGBAImage
Creates a latent of arbitrary color by encoding it with the provided VAE. Note that even though `0.5, 0.5, 0.5` seems like it should be equal to an empty latent, in reality it is not and seeds will be very different. Also comes in HSVA flavor.
  - Input
    - `r/g/b` : RGB in 0.0 -> 1.0 scale
    - `a` : Alpha. 0.0 for empty latent, 1.0 for entirely colored
    - `width/height/batch_size`: Same as EmptyLatentImage

#### BSZLatentGardient
Blend two latents together in a gradient pattern
  - Input
    - `a` : First latent; will copy params from this
    - `b` : Second latent
    - `pattern` : Gradient pattern
      - `sine` : Typical banded gradient. Both horizontal, vertial, diagonal depneding on frequency
      - `sine2` : Same as `sine` but the y axis is flipped
      - `circle` : Produces circles packed in a honeycomb shape
      - `squircle` : Produces squircles packed in a grid
      - `rings` : Produces rings centered on offsets. Circle but recursive instead of tiling
    - `xfrequency` : Pattern repetitions along X axis; horizontal gradient
    - `yfrequency` : Pattern repetitions along Y axis; vertical gradient
    - `xoffset` : Pattern start offset for X frequencies
    - `yoffset` : Pattern start offset for Y frequencies
    - `invert` : Invert A/B colors. Will still take params from A

#### BSZLatentHueChormaXL
Adjust an SDXL latent directly using Hue/Chroma/Lightness sliders.
Kind of works on non-XL latents, but not as accurately.
Input fields
  - `hue` : Hue offset in degrees
  - `chroma` : Multiply the chroma
  - `lightness` : Multiply the lightness

#### BSZLatentFill
Fill the four latent channels with arbitrary values.
  - Input
    - `latent` : Latent image(s).
    - `a/b/c/d` : Values for the four channels

#### BSZLatentDebug
Output information about the latent tensor into stdout

### bsz-pixelbuster.py
Nodes that require my own [Pixelbuster library](https://github.com/Beinsezii/pixelbuster).
Linux, Windows, and MacOS libraries are included directly in this node pack and won't have to be downloaded separately.

#### BSZPixelbuster
Write simple code to manipulate colors
Input fields
  - `image` : Image[s] to work on
  - `code` : Pixelbuster code. See [the help](https://github.com/Beinsezii/pixelbuster/blob/master/src/lib.rs#L10) for reference
  - `e1-e9` : Vars you can set externally that will be seen by the pixelbuster code as e1-e9

#### BSZLatentbuster
Write simple code to manipulate latent 'colors'
Input fields
  - `latent` : Latents[s] to work on. CIE LAB colorspace for the first 3 channels with the 4th being alpha.
  - `code` : Pixelbuster code. See [the help](https://github.com/Beinsezii/pixelbuster/blob/master/src/lib.rs#L10) for reference
  - `e1-e9` : Vars you can set externally that will be seen by the pixelbuster code as e1-e9

## Workflows

### sdxl.json
Personal flair of the SDXL "partial diffusion" workflow. Minimalist node setup with defaults balanced approach to speed/quality

#### Dependencies
  - `bsz-auto-hires.py` : While this workflow doesn't actually perform any upscaling, it still uses the `BSZAutoHiresCombined` node for quick aspect ratio changing and easy CLIP detail target adjustments

### sdxl-upscale.json
Personal flair of the SDXL "partial diffusion" workflow with added "High res fix". Slightly prioritizes speed as far as upscaling is concerned.

#### Dependencies
  - `bsz-auto-hires.py` : Workflow is painful without it.

### sdxl-principled.json
Complete demonstration of nodes in a compact workflow

#### Dependencies
Everything except `bsz-experimental.py`


## F.A.Q.
Question|Answer
---|---
What happened to the all-in-one XL node with upscaling and hires fix?|That design was inherently flawed due to ComfyUI's caching system. Now that ComfyUI has bypasses (CTRL+B) I re-wrote it into the single-stage chainable node you see now. This allows the same functionality while not having to restart the whole image when only changing the 2nd stage. Yes it's slightly messier looking but the time savings is huge. Please look at the `sdxl-principled.json` workflow for an example of how to most optimally use the new chained node.
Why is there a separate VAE loader instead of using the VAE directly from the main checkpoint?|I personally find it desireable to have the VAE decoupled from the checkpoint so you can change it without re-baking the models. If this isn't desirable to you yourself, simply remove the Load VAE node and reconnect the traces into the main Load Checkpoint node instead.
Why are the KSampler nodes so long?|To show live previews of each stage. I strongly recommend you do the same by launching ComfyUI with `--preview-method latent2rgb` or similar.
Why is *this* setting the default instead of *that* setting?|It just happens to look better on my benchmark images. If you think it's objectively wrong, open an issue with a compelling case on why it should be changed.
