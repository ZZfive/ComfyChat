# ComfyUI ResAdapter

**ComfyUI-ResAdapter** is an extension designed to enhance the usability of [ResAdapter](https://github.com/bytedance/res-adapter). It offers a simple node to load resadapter weights. Furthermore, this repo provide [specific workflows](examples) for text-to-image, accelerate-lora, controlnet and ip-adapter.

⭐ If ResAdapter is helpful to your images or projects, please help star this repo and [bytedance/res-adapter](https://github.com/bytedance/res-adapter). Thanks! 🤗

## Installation

```bash 
# Step1: Clone ComfyUI-ResAdapter
git clone git@github.com:jiaxiangc/ComfyUI-ResAdapter.git

# Step2: Move it to ComfyUI/custom_node
mv ComfyUI-ResAdapter ComfyUI/custom_nodes/

# Step3: Select Load ResAdapter Node in ComfyUI. We can use it like LoRA Loader.
```

## Download Model

We provide script for automatically downloading resadapter model from huggingface in `__init__.py`. If you can not connect with [huggingface.com](https://huggingface.co/), you can directly download model in [huggingface](https://huggingface.co/jiaxiangc/res-adapter). 

## Example Gallery

We provide workflow examples [here](examples).
There are demo videos for helping users to know how to use ComfyUI-ResAdapter, which is supported by [@fengyuzz](https://github.com/fengyuzzz).

### Text-to-Image

- workflow: [resadapter_text_to_image_workflow](examples/resadapter_text_to_image_workflow.json). 
- models: [dreamlike-diffusion-1.0](https://huggingface.co/dreamlike-art/dreamlike-diffusion-1.0)

https://github.com/jiaxiangc/ComfyUI-ResAdapter/assets/162297627/82453931-23de-4f72-8a9c-1053c4c8d81a

### ControlNet

- workflow: [resadapter_controlnet_workflow](examples/resadapter_controlnet_workflow.json). 
- models: [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5), [sd-controlnet-canny](https://huggingface.co/lllyasviel/sd-controlnet-canny)

https://github.com/jiaxiangc/ComfyUI-ResAdapter/assets/162297627/aab14d91-f68f-4415-b947-fd99662ca2ca

### IPAdapter

- workflow: [resadapter_ipadapter_workflow](examples/resadapter_ipadapter_workflow.json). 
- models: [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5), [IP-Adapter](https://huggingface.co/h94/IP-Adapter)

https://github.com/jiaxiangc/ComfyUI-ResAdapter/assets/162297627/92bcfc11-b1d5-4909-9e6d-7e1978c948c2

### Accelerate LoRA

- workflow: [resadapter_accelerate_lora_workflow](examples/resadapter_accelerate_lora_workflow.json). 
- models: [stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5), [lcm-lora-sdv1-5](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5)

https://github.com/jiaxiangc/ComfyUI-ResAdapter/assets/162297627/46ce4224-ab4d-4890-abc8-00b716dcb29a
