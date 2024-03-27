# ComfyUI_TravelSuite

ComfyUI custom nodes to apply various latent travel techniques.

## LatentTravel Node

Travel between different latent spaces using a range of blend and travel modes.

___**Node Inputs**___
- **A, B**: Latent variables needed for the process.
- **steps**: Integer representing the number of steps. This includes the two input latents A and B which will always be the first and last latents. The default value is 5. It should be within the range [0,10000].
- **factor**: A floating-point value with a default of 0.5. This value depends on the travel_mode specified
- **blend_mode**: Specifies the method used for latent blending: lerp", "slerp", "add", "multiply", "divide", "subtract", "overlay", "hard_light", "soft_light", "screen", "linear_dodge", "difference", "exclusion", "random"
- **travel_mode**: Determines the travel interval model:
- - 'linear': linear interpolation
- - 'hinge': takes factor as a cutpoint and splits the steps evenly between above and below the hinge 
- - 'circle': steps along the X-axis of a circle 
- - 'norm': steps along a normal distribution centered at 0.5. Factor is the gaussian scale factor.
- - 'quadratic', 'cubic', 'quartic': steps along linear space raised to 2, 3, 4 powers.  
- - 'geometric': steps along geomspace from 0, 1
- **reflect_travel**: Bool indicating whether to reflect travel mode values around the center. 
- **vae**: VAE (Variational Autoencoder) type, to decode generated latents to tensors. 
- **output_images**: Bool indicating whether to output images.  Requires VAE input. 
- **filepath**: String for defining the output path for the files. Default path is 'output/travel'.
- **prefix**: String for prefixing the generated files. By default, this is 'travel'. 
- **write_images**: Bool indicating whether to write output images.  Requires VAE input.

___**Node Outputs**___

- **LATENTS**: Resulting travel latents. There will be `steps` latents, and the first and last will be the input `A` and `B`.
- **IMAGES**: Tensor images, if `output_images=True`. Requires VAE.
- **FILEPATHS**: List of filepaths produced if `write_images=True`.


`[Workflow Image](workflow.png)`
