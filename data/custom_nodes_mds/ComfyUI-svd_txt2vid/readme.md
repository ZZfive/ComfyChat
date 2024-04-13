**Text to video for Stable Video Diffusion in ComfyUI**

This is node replaces the init_image conditioning for the [Stable Video Diffusion](https://github.com/Stability-AI/generative-models) image to video model with text embeds, together with a conditioning frame. The conditioning frame is a set of latents.

It is recommended to input the latents in a noisy state. Default ComfyUI noise does not create optimal results, so using other noise e.g. Power-Law Noise helps.

Motion bucket, fps & augmentation level preserved as possible input for the conditioning.

The video generation needs a couple frames time to get on it's feet and run.

Maxing out EDM sigma helps getting more colour into the image, together with additional prompts like "realistic". 

CFG can be raised way above normal parameters using VideoLinearCFGGuidance. Keeping a small CFG guidance spread of around 2 (e.g. min_cfg 18 & sampler cfg 20) helps with image quality consistency.
