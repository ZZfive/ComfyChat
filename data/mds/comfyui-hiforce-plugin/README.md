# Comfyui HiFORCE Plugin


Custom nodes pack provided by [HiFORCE](https://www.hiforce.net) for ComfyUI. This custom node helps to conveniently enhance images through Sampler, Upscaler, Mask, and more.

## NOTICE 
* You should install [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack). Many optimizations are built upon the foundation of ComfyUI-Impact-Pack.

## Installation

1. `cd custom_nodes`
2. `git clone https://github.com/hiforce/comfyui-hiforce-plugin.git`
3. `cd comfyui-hiforce-plugin` and run `pip install -r requirements.txt`


## Custom Nodes
### Samplers:
* **Basic Sampler:** Basic Sampler is very similar to the KSampler provided by ComfyUI, except that it exposes the 'full_drawing' option. This field corresponds to the 'return_with_leftover_noise' option in KSampler (Advanced), but with opposite values. The name 'full_drawing' is more user-friendly for developers familiar with the Stable Diffusion WebUI.
* **Loopback Sampler：** The Loopback Sampler allows you to generate images progressively using different drawing intensities.
* **HfTwoSamplersForMask：** Enhance of the TwoSamplersforMask of ComfyUI-Impact-Pack. We add enable option. If the 'enable' option is set to false, then this sampler will not function.
* **HfTwoStepSamplers:** HfTwoStepSamplers allows you to use different sampler algorithms and masks in two-step sampling.
* **HfIterativeLatentUpscale:** Enhance of the IterativeLatentUpscale of ComfyUI-Impact-Pack. We add enable option. If the 'enable' option is set to false, then this sampler will not function.
* **Some Swith Input for Sampler Node**

[Read More](https://github.com/hiforce/comfyui-hiforce-plugin/wiki/Sampler-Nodes-Introduction)

[Read More中文版](https://hiforce.yuque.com/org-wiki-hiforce-kbgemz/fpx22q/gtf57av9gkvgek5p)
