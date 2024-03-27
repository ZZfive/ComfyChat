# **Power Noise Suite** for [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

<img title="Power Noise Suite" src="https://github.com/WASasquatch/PowerNoiseSuite/assets/1151589/254600a1-1ae3-4560-99fb-b0ad6f36a713" width="400">

Power Noise Suite contains nodes centered around latent noise input, and diffusion, as well as latent adjustments. This repo is the successor to [PPF_Noise_ComfyUI](https://github.com/WASasquatch/PPF_Noise_ComfyUI) 

# What's new?

 - Power-Law Noise overhauled. Total revamp of the noise system was necessary for more accurate noise representation.
 - Tonal Guide Image. The Power KSampler now has a tonal guide image which can be used to help tone a generation similar to a input image. Not always reliable, but in some instances can theme the generation very much like the input image. 




[<img src="https://github.com/WASasquatch/PowerNoiseSuite/assets/1151589/b4a952f1-b040-4e5c-811c-b15b323d7426" width="400">](https://raw.githubusercontent.com/WASasquatch/ComfyUI_Workflows/main/Img2Img-Variations__TonalGuide.json)
<img src="https://github.com/WASasquatch/PowerNoiseSuite/assets/1151589/23bb0f9b-4f63-41ea-961d-ac19e554feef" width="400">

## Workflows
 - [Img2Img Variations with Tonal Guide](https://raw.githubusercontent.com/WASasquatch/ComfyUI_Workflows/main/Img2Img-Variations__TonalGuide.json)


---

# Installation
 - Clone the repo to `ComfyUI/custom_nodes`.

---

# Nodage Menu
 - [Perlin Power Fractal Noise](https://github.com/WASasquatch/PowerNoiseSuite/#perlin-power-fractal-noise-parameters)
 - [Power Law Noise](https://github.com/WASasquatch/PowerNoiseSuite/#power-law-noise-parameters)
 - [Cross-Hatch Power Fractal](https://github.com/WASasquatch/PowerNoiseSuite/#cross-hatch-power-fractal-parameters)
 - [Cross-Hatch Linear Power Fractal](https://github.com/WASasquatch/PowerNoiseSuite/#cross-hatch-linear-power-fractal-parameters)
 - [Blend Latents](https://github.com/WASasquatch/PowerNoiseSuite/#blend-latents-parameters)
 - [Image as Latents](https://github.com/WASasquatch/PowerNoiseSuite/#images-as-latents-parameters)
 - [Latent Adjustment](https://github.com/WASasquatch/PowerNoiseSuite/#latent-adjustment-parameters)
 - [Power KSampler Advanced](https://github.com/WASasquatch/PowerNoiseSuite/#power-ksampler-advanced-parameters)
   - [Perlin Power Fractal Settings](https://github.com/WASasquatch/PowerNoiseSuite/#perlin-power-fractal-settings-parameters)
   - [Cross-Hatch Power Fractal Settings](https://github.com/WASasquatch/PowerNoiseSuite/#cross-hatch-power-fractal-settings-parameters)


---

# Nodes

## **Perlin Power Fractal Noise** Parameters

This node generates perlin power fractal noise

### Required:

- `batch_size` (int): Number of noisy tensors to generate in the batch.
    - Range: [1, 64]
- `width` (int): Width of each tensor in pixels.
    - Range: [64, 8192]
- `height` (int): Height of each image in pixels.
- `resampling` (string): This parameter determines the resampling method used for scaling noise to the latent size. Choose from the following options:
    - "**nearest-exact**": Nearest-Exact Resampling:
        - Nearest-neighbor resampling selects the pixel value from the nearest source pixel, resulting in a blocky, pixelated appearance. It preserves the exact values without interpolation.
    - "**bilinear**": Bilinear Resampling:
        - Bilinear interpolation takes a weighted average of the four nearest source pixels, producing smoother transitions between pixels. It's a good choice for general image resizing.
    - "**area**": Area Resampling (Antialiasing):
        - Resampling using pixel area relation, also known as antialiasing, computes pixel values based on the areas of contributing source pixels. It reduces aliasing artifacts and is suitable for preserving fine details.
    - "**bicubic**": Bicubic Resampling:
        - Bicubic interpolation uses a cubic polynomial to compute pixel values based on the 16 nearest source pixels. It provides smoother transitions and better detail preservation, suitable for high-quality resizing.
    - "**bislerp**": Bislerp Resampling (Bilinear Sinc Interpolation):
        - Bislerp interpolation combines bilinear simplicity with sinc function interpolation, resulting in high-quality resizing with reduced artifacts. It offers a balance between quality and computational cost.
- `X` (float): X-coordinate offset for noise sampling.
    - Range: [-99999999, 99999999]
- `Y` (float): Y-coordinate offset for noise sampling.
    - Range: [-99999999, 99999999]
- `Z` (float): Z-coordinate offset for noise sampling.
    - Range: [-99999999, 99999999]
- `frame` (int): The current frame number for time evolution.
    - Range: [0, 99999999]
- `evolution_factor` (float): Factor controlling time evolution. Determines how much the noise evolves over time based on the batch index.
    - Range: [0.0, 1.0]
- `octaves` (int): Number of octaves for fractal generation. Controls the level of detail and complexity in the output.
    - Range: [1, 8]
- `persistence` (float): Persistence parameter for fractal generation. Determines the amplitude decrease of each octave.
    - Range: [0.01, 23.0]
- `lacunarity` (float): Lacunarity parameter for fractal generation. Controls the increase in frequency from one octave to the next.
    - Range: [0.01, 99.0]
- `exponent` (float): Exponent applied to the noise values. Adjusting this parameter controls the overall intensity and contrast of the output.
    - Range: [0.01, 38.0]
- `scale` (float): Scaling factor for frequency of noise. Larger values produce smaller, more detailed patterns, while smaller values create larger patterns.
    - Range: [2, 2048]
- `brightness` (float): Adjusts the overall brightness of the generated noise.
    - -1.0 makes the noise completely black.
    - 0.0 has no effect on brightness.
    - 1.0 makes the noise completely white.
    - Range: [-1.0, 1.0]
- `contrast` (float): Adjusts the contrast of the generated noise.
    - -1.0 reduces contrast, enhancing the difference between dark and light areas.
    - 0.0 has no effect on contrast.
    - 1.0 increases contrast, enhancing the difference between dark and light areas.
    - Range: [-1.0, 1.0]
- `clamp_min` (float): The floor range of the noise
  - Range: [-10.0, 10]
- `clamp_max` (float): The ceiling range of the noise
 - Range: [-10, 10]
- `seed` (int, optional): Seed for random number generation. If None, uses random seeds for each batch.
    - Range: [0, 0xffffffffffffffff]
- `device` (string): Specify the device to generate noise on, either "cpu" or "cuda".

### Optional:

- `optional_vae` (VAE, optional): The optional VAE for encoding the noise.

### Returns

- `latents` (LATENT): The latent image result
- `previews` (IMAGE): The noise as tensor images

---

## Power Law Noise Parameters

This node generates Power-Law noise. Power law noise is a common form of noise used all over. For example, `vanilla_comfyui` mode is regular ComfyUI noise that is White Noise.

### Input Types

This class provides the following input parameters:

- `batch_size` (INT): The batch size for generating noise.
  - Default: 1
  - Range: [1, 64]
  - Step: 1
- `width` (INT): The width of the generated noise image.
  - Default: 512
  - Range: [64, 8192]
  - Step: 1
- `height` (INT): The height of the generated noise image.
  - Default: 512
  - Range: [64, 8192]
  - Step: 1
- `resampling` (List of Strings): The resampling method to use for resizing.
  - Options: ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
- `noise_type`: ["white", "grey", "pink", "green", "blue", "mix"]: 
  - The type of [power-law noise](https://en.wikipedia.org/wiki/Colors_of_noise#Technical_definitions) to generate.
- `frequency` (FLOAT): Frequency of the power law noise.
  - Default: 64
  - Range: [0.001, 1024.0]
  - Step: 0.001
- `attenuation` (FLOAT): Attenuation factor for the power law noise.
  - Default: 1.0
  - Range: [0.001, 1024.0]
  - Step: 0.001
- `seed` (INT): Seed value for random number generation.
  - Default: 0
  - Range: [0, 18446744073709551615]
- `device` (List of Strings): The device on which to generate the noise.
  - Options: ["cpu", "cuda"]
- `optional_vae` (VAE): An optional Variational Autoencoder (VAE) instance.
  - Default: None

### Returns

- `latents` (LATENT): The latent image result
- `previews` (IMAGE): The noise as tensor images

---

## **Cross-Hatch Power Fractal** Parameters

This node generates a batch of cross-hatch power fractal noise patterns.

### Required:

- `batch_size` (int): Number of noisy tensors to generate in the batch.
    - Range: [1, 64]
- `width` (int): Width of each tensor in pixels.
    - Range: [64, 8192]
- `height` (int): Height of each image in pixels.
- `resampling` (string): This parameter determines the resampling method used for scaling noise to the latent size. Choose from the following options:
    - "**nearest-exact**": Nearest-Exact Resampling:
        - Nearest-neighbor resampling selects the pixel value from the nearest source pixel, resulting in a blocky, pixelated appearance. It preserves the exact values without interpolation.
    - "**bilinear**": Bilinear Resampling:
        - Bilinear interpolation takes a weighted average of the four nearest source pixels, producing smoother transitions between pixels. It's a good choice for general image resizing.
    - "**area**": Area Resampling (Antialiasing):
        - Resampling using pixel area relation, also known as antialiasing, computes pixel values based on the areas of contributing source pixels. It reduces aliasing artifacts and is suitable for preserving fine details.
    - "**bicubic**": Bicubic Resampling:
        - Bicubic interpolation uses a cubic polynomial to compute pixel values based on the 16 nearest source pixels. It provides smoother transitions and better detail preservation, suitable for high-quality resizing.
    - "**bislerp**": Bislerp Resampling (Bilinear Sinc Interpolation):
        - Bislerp interpolation combines bilinear simplicity with sinc function interpolation, resulting in high-quality resizing with reduced artifacts. It offers a balance between quality and computational cost.
- `frequency` (float): Frequency parameter for fractal generation. Determines the frequency of the cross-hatch pattern.
    - Range: [0.001, 1024.0]
- `octaves` (int): Number of octaves for fractal generation. Controls the level of detail and complexity in the output.
    - Range: [1, 32]
- `persistence` (float): Persistence parameter for fractal generation. Determines the amplitude decrease of each octave.
    - Range: [0.001, 2.0]
- `color_tolerance` (float): Tolerance parameter for color mapping. Affects the variety of colors in the output.
    - Range: [0.001, 1.0]
- `num_colors` (int): Number of colors to use in the output.
    - Range: [2, 256]
- `angle_degrees` (float): Angle in degrees for the cross-hatch pattern.
    - Range: [0.0, 360.0]
- `brightness` (float): Adjusts the overall brightness of the generated noise.
    - -1.0 makes the noise completely black.
    - 0.0 has no effect on brightness.
    - 1.0 makes the noise completely white.
    - Range: [-1.0, 1.0]
- `contrast` (float): Adjusts the contrast of the generated noise.
    - -1.0 reduces contrast, enhancing the difference between dark and light areas.
    - 0.0 has no effect on contrast.
    - 1.0 increases contrast, enhancing the difference between dark and light areas.
    - Range: [-1.0, 1.0]
- `blur` (float): Blur parameter for the generated noise.
    - Range: [0.0, 1024.0]
- `clamp_min` (float): The floor range of the noise.
    - Range: [-10.0, 10.0]
- `clamp_max` (float): The ceiling range of the noise.
    - Range: [-10.0, 10.0]
- `seed` (int, optional): Seed for random number generation. If None, uses random seeds for each batch.
    - Range: [0, 0xffffffffffffffff]
- `device` (string): Specify the device to generate noise on, either "cpu" or "cuda".

### Optional:

- `optional_vae` (VAE, optional): The optional VAE for encoding the noise.

### Returns
- `latents` (LATENT): The latent image result
- `previews` (IMAGE): The noise as tensor images.

---

## **Cross-Hatch Linear Power Fractal** Parameters

This node generates a batch of *linear* cross-hatch power fractal noise patterns.

### Required Parameters:

- `batch_size` (INT): The number of images to generate in the batch.
    - Range: [1, 64]
    - Default: 1
- `width` (INT): The width of each image in pixels.
    - Range: [64, 8192]
    - Default: 512
- `height` (INT): The height of each image in pixels.
    - Range: [64, 8192]
    - Default: 512
- `resampling` (STRING): The resampling method used for scaling noise to the latent size. Choose from the following options:
    - "**nearest-exact**": Nearest-Exact Resampling.
    - "**bilinear**": Bilinear Resampling.
    - "**area**": Area Resampling (Antialiasing).
    - "**bicubic**": Bicubic Resampling.
    - "**bislerp**": Bislerp Resampling (Bilinear Sinc Interpolation).
- `frequency` (FLOAT): The frequency parameter for fractal generation. It determines the frequency of the cross-hatch pattern.
    - Range: [0.001, 1024.0]
    - Default: 320.0
- `gain` (FLOAT): The gain factor for controlling the amplitude of the generated noise.
    - Range: [0.0, 1.0]
    - Default: 0.25
- `octaves` (INT): The number of octaves for fractal generation. It controls the level of detail and complexity in the output.
    - Range: [1, 32]
    - Default: 12
- `persistence` (FLOAT): The persistence parameter for fractal generation. It determines the amplitude decrease of each octave.
    - Range: [0.001, 2.0]
    - Default: 1.5
- `add_noise` (FLOAT): The tolerance parameter for adding random noise to the generated pattern. It affects the variety of colors in the output.
    - Range: [0.0, 1.0]
    - Default: 0.0
- `linear_range` (INT): The range for mapping noise values to different levels.
    - Range: [2, 256]
    - Default: 16
- `linear_tolerance` (FLOAT): The tolerance parameter for color mapping.
    - Range: [0.001, 1.0]
    - Default: 0.05
- `angle_degrees` (FLOAT): The angle in degrees for the cross-hatch pattern.
    - Range: [0.0, 360.0]
    - Default: 45.0
- `brightness` (FLOAT): Adjusts the overall brightness of the generated noise.
    - Range: [-1.0, 1.0]
    - Default: 0.0
- `contrast` (FLOAT): Adjusts the contrast of the generated noise.
    - Range: [-1.0, 1.0]
    - Default: 0.0
- `seed` (INT): The seed for random number generation.
    - Range: [0, 0xffffffffffffffff]
    - Default: 0
- `device` (STRING): Specify the device to generate noise on, either "cpu" or "cuda".

### Optional Parameters:

- `optional_vae` (VAE): The optional VAE for encoding the noise.

### Returns
- `latents` (LATENT): The latent image result
- `previews` (IMAGE): The noise as tensor images.

---

## **Blend Latents** Parameters

This node provides a method for blending two latent images.

### Required:

- `latent_a` (LATENT, required): The first input latent image to be blended.
- `latent_b` (LATENT, required): The second input latent image to be blended.
- `operation` (string, required): The blending operation to apply. Choose from the following options:
  - **add**: Combines two images by adding their pixel values together. 
  - **bislerp**: Interpolates between two images smoothly using the factor `t`.
  - **color dodge**: Brightens the base image based on the blend image. It creates a high-contrast effect by making bright areas of the blend image affect the base image more.
  - **cosine interp**: Interpolates between two images using a cosine function.
  - **cuberp**: Blends two images by applying a cubic interpolation.
  - **difference**: Subtracts one image from another and takes the absolute value. It highlights the differences between the two images.
  - **exclusion**: Combines two images using an exclusion formula, resulting in a unique contrast effect.
  - **glow**: Create a glow effect based on the blend image. Similar to pin light, but darker. 
  - **hard light**: Combines two images in a way that enhances the contrast. It creates a sharp transition between light and dark areas.
  - **lerp**: Linearly interpolates between two images based on the factor. It creates a simple linear transition.
  - **linear dodge**: Brightens the base image by adding the blend image, creating a high-key effect.
  - **linear light**: Blends two images to enhance contrast. It brightens or darkens the base image based on the blend image.
  - **multiply**: Multiplies the pixel values of two images, resulting in a darker image with increased contrast.
  - **overlay**: Combines two images using an overlay formula. It enhances the contrast and creates a dramatic effect.
  - **pin light**: Combines two images in a way that preserves the details and intensifies the colors.
  - **random**: Adds random noise to both images, creating a noisy and textured effect.
  - **reflect**: Combines two images in a reflection formula. A interesting blend to say the least. 
  - **screen**: Brightens the base image based on the blend image, resulting in a high-key effect.
  - **slerp**: Spherically interpolates between two images, creating a smooth and curved transition.
  - **subtract**: Subtracts the blend image from the base image, resulting in a darker image.
  - **vivid light**: Enhances the vividness of colors in the base image based on the blend image. It intensifies the colors.
- `blend_ratio` (FLOAT, required): The blend ratio between `latent_a` and `latent_b`. 
    - Default: 0.5
    - Range: [0.01, 1.0]
- `blend_strength` (FLOAT, required): The strength of the blending operation.
    - Default: 1.0
    - Range: [0.0, 100.0]
	
### Optional:

- `mask` (MASK, optional): An optional mask tensor to control the blending region.
- `set_noise_mask` (string, optional): Whether to set the noise mask. Choose from "false" or "true".
- `normalize` (string, optional): Whether to normalize the resulting latent image. Choose from "false" or "true".
- `clamp_min` (FLOAT, optional): The minimum clamping range for the output.
    - Default: 0.0
    - Range: [-10.0, 10.0]
- `clamp_max` (FLOAT, optional): The maximum clamping range for the output.
    - Default: 1.0
    - Range: [-10.0, 10.0]

### Returns

- `latent` (LATENT): The latent image result.

---

## **Images as Latents** Parameters

This node converts `IMAGE` to `LATENT` format, without encoding them. Really only useful for raw noise.

### Required:
- `images` (IMAGE): Input images to be converted into latent images.
- `resampling` (string): This parameter determines the resampling method used for scaling images to the latent size. Choose from the following options:
    - "**nearest-exact**": Nearest-Exact Resampling:
        - Nearest-neighbor resampling selects the pixel value from the nearest source pixel, resulting in a blocky, pixelated appearance. It preserves the exact values without interpolation.
    - "**bilinear**": Bilinear Resampling:
        - Bilinear interpolation takes a weighted average of the four nearest source pixels, producing smoother transitions between pixels. It's a good choice for general image resizing.
    - "**area**": Area Resampling (Antialiasing):
        - Resampling using pixel area relation, also known as antialiasing, computes pixel values based on the areas of contributing source pixels. It reduces aliasing artifacts and is suitable for preserving fine details.
    - "**bicubic**": Bicubic Resampling:
        - Bicubic interpolation uses a cubic polynomial to compute pixel values based on the 16 nearest source pixels. It provides smoother transitions and better detail preservation, suitable for high-quality resizing.
    - "**bislerp**": Bislerp Resampling (Bilinear Sinc Interpolation):
        - Bislerp interpolation combines bilinear simplicity with sinc function interpolation, resulting in high-quality resizing with reduced artifacts. It offers a balance between quality and computational cost.

### Returns
- `tuple` (LATENT, IMAGE): A tuple containing the generated latent image and the input images.

---

## **Latent Adjustment** Parameters

This node allows your to adjust the latents such as brightness, contrast, and sharpness. 

### Required Inputs

- `latents` (LATENT): The input latent image to be adjusted.
- `brightness` (FLOAT): Adjusts the brightness of the latent image.
    - Default: 1.0
    - Range: -1.0 to 2.0
    - Step: 0.001
- `contrast` (FLOAT): Adjusts the contrast of the latent image.
    - Default: 1.0
    - Range: -1.0 to 2.0
    - Step: 0.001
- `saturation` (FLOAT): Adjusts the saturation of the latent image.
    - Default: 1.0
    - Range: 0.0 to 2.0
    - Step: 0.001
- `exposure` (FLOAT): Adjusts the exposure of the latent image.
    - Default: 0.0
    - Range: -1.0 to 2.0
    - Step: 0.001
- `alpha_sharpen` (FLOAT): Applies alpha sharpening to the latent image.
    - Default: 0.0
    - Range: 0.0 to 10.0
    - Step: 0.01
- `high_pass_radius` (FLOAT): Applies high-pass filtering to the latent image.
    - Default: 0.0
    - Range: 0.0 to 1024.0
    - Step: 0.01
- `high_pass_strength` (FLOAT): Specifies the strength of the high-pass filtering.
    - Default: 1.0
    - Range: 0.0 to 2.0
    - Step: 0.01
- `clamp_min` (FLOAT): Sets the minimum clamping range for the output.
    - Default: 0.0
    - Range: -10.0 to 10.0
    - Step: 0.01
- `clamp_max` (FLOAT): Sets the maximum clamping range for the output.
    - Default: 1.0
    - Range: -10.0 to 10.0
    - Step: 0.01

### Optional Inputs

- `latent2rgb_preview` (BOOL): Convert the latent (if encoded) to a RGB representation (not very accurate).
    - Default: false

## Returns

- `latents` (LATENT): The latent image result
- `previews` (IMAGE): The noise as tensor images

---

## **Power KSampler Advanced** Parameters

The `PPFNKSamplerAdvanced` class is part of the Power Noise Suite and provides advanced capabilities for noise sampling. It is categorized under "Power Noise Suite/Sampling."

#### Notes:
- Where are all the samplers? Power KSampler Advanced only worked with samplers that inject noise on every step, and utilize the `comfy.k_diffusion.sampling.default_noise_sampler()` method.
- Outputes tinted green or just plain non-sense? Your tolerance may be too high. Additionally using a input latent with strong starting noise can result in this as well. 
 - No that's not the reason? Well then your model, and scheduelr selection is producing an abornmally high `sigma` value that is throwing off the sigma scaling. 

### Required Inputs

- `model` (MODEL): The model used for sampling.
- `add_noise` (["enable", "disable"]): Specifies whether to add noise during sampling.
    - Default: "enable"
- `seed` (INT): The random seed for sampling.
    - Default: 0
    - Range: 0 to 18,446,744,073,709,551,615
- `steps` (INT): The number of total sampling steps.
    - Default: 20
    - Range: 1 to 10,000
- `cfg` (FLOAT): Classifier Free Guidance scale (higher values *try* to adhere closer to your prompt)
    - Default: 8.0
    - Range: 0.0 to 100.0
- `sampler_name` (sampler_name): The name of the sampler to user. *Power KSampler Advanced has a limited set of samplers that it is compatible with.*
- `scheduler` (SCHEDULERS): The scheduler used by sampling.
- `positive` (CONDITIONING): Conditioning prompt.
- `negative` (CONDITIONING): Conditioning prompt.
- `latent_image` (LATENT): The latent image.
- `start_at_step` (INT): Sampling starting step.
    - Default: 0
    - Range: 0 to 10,000
- `end_at_step` (INT): Sampling ending step.
    - Default: 10,000
    - Range: 0 to 10,000
- `enable_denoise` ["false", "true"]
    -  Whether to enable the use of the denoise value rather then `start_at_step` and `end_at_step`. If enabled, `start_at_step` and `end_at_step` will be set to `None`.
- `return_with_leftover_noise` (["disable", "enable"]): Specifies whether to return the latent image with left of noise *(only really helpful for feeding another KSampler advanced)*
- `denoise` (FLOAT): The denoising value for img2img inference.
    - Default: 1.0
    - Range: 0 to 1.0
    - Step: 0.01
### Optional Inputs

- `noise_type`: ["white", "grey", "pink", "green", "blue", "random_mix", "brownian", "velvet", "violet", "vanilla_comfy"]: 
  - The type of [power-law noise](https://en.wikipedia.org/wiki/Colors_of_noise#Technical_definitions) to generate.
- `noise_blending`: ["bislerp", "cosine interp", "cuberp", "hslerp", "lerp", "add", "inject"]
  - The noise blending method used during sampling
- `noise_mode`: ["additive", "subtractive"]
  - The noise operation used on the Power noise added to base noise.
- `scale` (FLOAT): The scale of the noise range.
    - Default: 1.0
    - Range: 0.001 to [sys.maxsize-1]
    - Step: 0.001
- `alpha_exponent` (FLOAT): The noise alpha exponent for Power-Law Noise. Higher it is, the softer noise will be.
    - Default: 1.0
    - Range: 0.001 to 12.0
    - Step: 0.001
- `modulator` (FLOAT): Modulator used for Brownian noise. Changes noise density.
    - Default: 1.0
    - Range: 0.001 to 12.0
    - Step: 0.001
- `sigma_tolerance` (FLOAT): The sigma tolerance controls how the amplitude of sigma to apply noise with. 
    - Default: 0.5
    - Range: 0.0 to 1.0
    - Step: 0.001
- `ppf_settings` (PPF_SETTINGS): Settings for power fractal noise. If plugged, Power-Law, and Cross-Hatch noise will be disabled.
- `ch_settings` (CH_SETTINGS): Settings for cross-hatch noise. If plugged, Power-Law noise will be disabled. `ppf_setttings` will bypass this input.
- `tonal_guide_latent` (CH_SETTINGS): An optional latent image used to potentially help guide the colors and sometimes influences composition. 
- `guide_use_noise` ["true", "false"]
 - Whether to use Power-Law noise with latent guide. 

### Returns

- `latents` (LATENT): The latent image result.

---

## **Perlin Power Fractal Settings** Parameters

Define Perlin Power Fractal settings for Power KSampler Advanced

### Input Types

- `X` (FLOAT): X-coordinate offset for noise sampling.
  - Default: 0
  - Range: [-99999999, 99999999]
  - Step: 0.01
- `Y` (FLOAT): Y-coordinate offset for noise sampling.
  - Default: 0
  - Range: [-99999999, 99999999]
  - Step: 0.01
- `Z` (FLOAT): Z-coordinate offset for noise sampling.
  - Default: 0
  - Range: [-99999999, 99999999]
  - Step: 0.01
- `evolution` (FLOAT): Factor controlling time evolution. Determines how much the noise evolves over time.
  - Default: 0.0
  - Range: [0.0, 1.0]
  - Step: 0.01
- `frame` (INT): The current frame number for time evolution.
  - Default: 0
  - Range: [0, 99999999]
  - Step: 1
- `scale` (FLOAT): Scaling factor for frequency of noise. Larger values produce smaller, more detailed patterns, while smaller values create larger patterns.
  - Default: 5
  - Range: [2, 2048]
  - Step: 0.01
- `octaves` (INT): Number of octaves in the fractal noise.
  - Default: 8
  - Range: [1, 8]
  - Step: 1
- `persistence` (FLOAT): Persistence of the fractal noise.
  - Default: 1.5
  - Range: [0.01, 23.0]
  - Step: 0.01
- `lacunarity` (FLOAT): Lacunarity of the fractal noise.
  - Default: 2.0
  - Range: [0.01, 99.0]
  - Step: 0.01
- `exponent` (FLOAT): Exponent of the fractal noise.
  - Default: 4.0
  - Range: [0.01, 38.0]
  - Step: 0.01
- `brightness` (FLOAT): Brightness adjustment for the generated noise.
  - Default: 0.0
  - Range: [-1.0, 1.0]
  - Step: 0.01
- `contrast` (FLOAT): Contrast adjustment for the generated noise.
  - Default: 0.0
  - Range: [-1.0, 1.0]
  - Step: 0.01

### Returns

- `ppf_settings` (PPF_SETTINGS): Dictionary settings for Power KSampler Advanced.

---

## **Cross-Hatch Power Fractal Settings** Parameters

Define Cross-Hatch Power Fractal settings for Power KSampler Adanced.

### Input Types

- `frequency` (FLOAT): Frequency of the cross-hatch pattern.
  - Default: 320.0
  - Range: [0.001, 1024.0]
  - Step: 0.001
- `octaves` (INT): Number of octaves in the cross-hatch noise.
  - Default: 12
  - Range: [1, 32]
  - Step: 1
- `persistence` (FLOAT): Persistence of the cross-hatch noise.
  - Default: 1.5
  - Range: [0.001, 2.0]
  - Step: 0.001
- `num_colors` (INT): Number of colors in the generated cross-hatch pattern.
  - Default: 16
  - Range: [2, 256]
  - Step: 1
- `color_tolerance` (FLOAT): Color tolerance for the cross-hatch pattern.
  - Default: 0.05
  - Range: [0.001, 1.0]
  - Step: 0.001
- `angle_degrees` (FLOAT): Angle in degrees for the cross-hatch pattern.
  - Default: 45.0
  - Range: [0.0, 360.0]
  - Step: 0.01
- `brightness` (FLOAT): Brightness adjustment for the generated cross-hatch pattern.
  - Default: 0.0
  - Range: [-1.0, 1.0]
  - Step: 0.001
- `contrast` (FLOAT): Contrast adjustment for the generated cross-hatch pattern.
  - Default: 0.0
  - Range: [-1.0, 1.0]
  - Step: 0.001
- `blur` (FLOAT): Blur amount for the generated cross-hatch pattern.
  - Default: 2.5
  - Range: [0, 1024]
  - Step: 0.01

### Returns

- `ch_settings` (CH_SETTINGS): Dictionary settings for Power KSampler Advanced.
