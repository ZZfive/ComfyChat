Auto-MBW for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) loosely based on [sdweb-auto-MBW](https://github.com/Xerxemi/sdweb-auto-MBW)

### Purpose
This node "advanced > auto merge block weighted" takes two models, merges individual blocks together at various ratios, and automatically rates each merge, keeping the ratio with the highest score. Whether this is a good idea or not is anyone's guess. In practice this makes models that make images the classifier says are good. You would probably disagree with the classifiers' decisions often.

### Settings
- Prompt: to generate sample images to be rated
- Sample Count: number of samples per ratio per block to generate
- Search Depth: number of branches to take while choosing ratios to test
- Classifier: model used to rate images

### Search Depth
To calculate ratios to test, the node branches out from powers of 0.5 

- A depth of 2 will examine 0.0, 0.5, 1.0 
- A depth of 4 will examine 0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0 
- A depth of 6 will examine 33 different ratios

There are 25 blocks to examine. If you use a depth of 4 and create 2 samples each, `25 * 9 * 2 = 450` images will be generated.

### Classifier
The classifier models have been taken from the sdweb-auto-MBW repo.

- [Laion Aesthetic Predictor](https://huggingface.co/spaces/Geonmo/laion-aesthetic-predictor)
- [Waifu Diffusion 1.4 aesthetic model](https://huggingface.co/hakurei/waifu-diffusion-v1-4)
- [Cafe Waifu](https://huggingface.co/cafeai/cafe_waifu) and [Cafe Aesthetic](https://huggingface.co/cafeai/cafe_aesthetic)

### Notes
- many hardcoded settings are arbitrary such as the seed, sampler and block processing order
- generated images are not saved
- the resulting model will contain the text encoder and VAE sent to the node

### Bugs
- merging process doesn't use the comfy ModelPatcher method and takes hundreds of milliseconds
- - as a result, --highvram flag recommended. both models will be kept in VRAM and the process is much faster
- the unet will (probably) be fp16 and the rest fp32. that's how they're sent to the node
- - see: `model_management.should_use_fp16()`
