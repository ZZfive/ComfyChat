
# Currently supported Custom Nodes and Models

Below a list of resources used on our platform that we are currently compatible with. This page will likely receive frequent updates. If there is a node suite that's necessary for a workflow, you can leave feedback [at our Discord](https://discord.gg/saltai)

- [Custom Nodes](#custom_nodes)
- [Models](#models)

# Custom_Nodes

| Name                                         | Repository                                                              | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|----------------------------------------------|-------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ComfyMath                                    | [https://github.com/evanspearman/ComfyMath](https://github.com/evanspearman/ComfyMath)                                                 | Provides Math Nodes for ComfyUI. Boolean Logic, Integer Arithmetic, Floating Point Arithmetic and Functions, Vec2, Vec3, and Vec4 Arithmetic and Functions                                                                                                                                                                                                                                                                                                                                                |
| ComfyUI-Advanced-ControlNet                  | [https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet)               | Nodes: ControlNetLoaderAdvanced, DiffControlNetLoaderAdvanced, ScaledSoftControlNetWeights, SoftControlNetWeights, CustomControlNetWeights, SoftT2IAdapterWeights, CustomT2IAdapterWeights                                                                                                                                                                                                                                                                                                            |
| AnimateDiff Evolved                          | [https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved )              | A forked repository that actively maintains [a/AnimateDiff](https://github.com/ArtVentureX/comfyui-animatediff), created by ArtVentureX. Improved AnimateDiff integration for ComfyUI, adapts from sd-webui-animatediff. [w/Download one or more motion models from [a/Original Models](https://huggingface.co/guoyww/animatediff/tree/main) | [a/Finetuned Models](https://huggingface.co/manshoety/AD_Stabilized_Motion/tree/main). See README for additional model links and usage. Put the model weights under %%ComfyUI/custom_nodes/ComfyUI-AnimateDiff-Evolved/models%%. You are free to rename the models, but keeping original names will ease use when sharing your workflow.]     |
| pythongosssss/ComfyUI-Custom-Scripts         | [https://github.com/pythongosssss/ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts)                     | This extension provides: Auto Arrange Graph, Workflow SVG, Favicon Status, Image Feed, Latent Upscale By, Lock Nodes & Groups, Lora Subfolders, Preset Text, Show Text, Touch Support, Link Render Mode, Locking, Node Finder, Quick Nodes, Show Image On Menu, Show Text, Workflow Managements, Custom Widget Default Values                                                                                                                                                                       |
| ComfyUI Frame Interpolation                  | [https://github.com/Fannovel16/ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation)                 | Nodes: KSampler Gradually Adding More Denoise (efficient)                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| ComfyUI Impact Pack                          | [https://github.com/ltdrdata/ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)                                     | This extension offers various detector nodes and detailer nodes that allow you to configure a workflow that automatically enhances facial details. And provide iterative upscaler. [w/NOTE:'Segs & Mask' has been renamed to 'ImpactSegsAndMask.' Please replace the node with the new name.]                                                                                                                                                                                                               |
| KJNodes for ComfyUI                          | [https://github.com/kijai/ComfyUI-KJNodes](https://github.com/kijai/ComfyUI-KJNodes)                                                   | Various quality of life -nodes for ComfyUI, mostly just visual stuff to improve usability.                                                                                                                                                                                                                                                                                                                                                                                                                |
| ComfyUI-VideoHelperSuite                     | [https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)                     | Nodes: VHS_VideoCombine. Nodes related to video workflows                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| Advanced CLIP Text Encode                    | [https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb](https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb)                             | Advanced CLIP Text Encode (if you need A1111 like prompt. you need this. But Cutoff node includes this feature, already.)                                                                                                                                                                                                                                                                                                                                                                                |
| ComfyUI_Comfyroll_CustomNodes                | [https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes](https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes)                     | Custom nodes for SDXL and SD1.5 including Multi-ControlNet, LoRA, Aspect Ratio, Process Switches, and many more nodes. NOTE: Maintainer is changed to Suzie1 from RockOfFire. [w/Using an outdated version has resulted in reported issues with updates not being applied. Trying to reinstall the software is advised.]                                                                                                                                                        |
| FizzNodes                                    | [https://github.com/FizzleDorf/ComfyUI_FizzNodes](https://github.com/FizzleDorf/ComfyUI_FizzNodes)                                     | Scheduled prompts, scheduled float/int values and wave function nodes for animations and utility. compatable with [a/framesync](https://www.framesync.xyz/) and [a/keyframe-string-generator](https://www.chigozie.co.uk/keyframe-string-generator/) for audio synced animations in Comfyui.                                                                                                                                                                                                               |
| ComfyUI_IPAdapter_plus                       | [https://github.com/cubiq/ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus)                                     | ComfyUI reference implementation for IPAdapter models. The code is mostly taken from the original IPAdapter repository and laksjdjf's implementation, all credit goes to them. I just made the extension closer to ComfyUI philosophy.                                                                                                                                                                                                                                                                    |
| ComfyUI_experiments                          | [https://github.com/comfyanonymous/ComfyUI_experiments](https://github.com/comfyanonymous/ComfyUI_experiments)                         | Nodes: ModelSamplerTonemapNoiseTest, TonemapNoiseWithRescaleCFG, ReferenceOnlySimple, RescaleClassifierFreeGuidanceTest, ModelMergeBlockNumber, ModelMergeSDXL, ModelMergeSDXLTransformers, ModelMergeSDXLDetailedTransformers.[w/NOTE: This is a consolidation of the previously separate custom nodes. Please delete the sampler_tonemap.py, sampler_rescalecfg.py, advanced_model_merging.py, sdxl_model_merging.py, and reference_only.py files installed in custom_nodes before.]   |
| Comfy_KepListStuff                           | [https://github.com/M1kep/Comfy_KepListStuff](https://github.com/M1kep/Comfy_KepListStuff)                                             | Nodes: Range(Step), Range(Num Steps), List Length, Image Overlay, Stack Images, Empty Images, Join Image Lists, Join Float Lists. This extension provides various list manipulation nodes                                                                                                                                                                                                                                                                                                                 |
| Derfuu_ComfyUI_ModdedNodes                   | [https://github.com/Derfuu/Derfuu_ComfyUI_ModdedNodes](https://github.com/Derfuu/Derfuu_ComfyUI_ModdedNodes)                           | Automate calculation depending on image sizes or something you want.                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| WAS_Extras                                   | [https://github.com/WASasquatch/WAS_Extras](https://github.com/WASasquatch/WAS_Extras)                                                 | Nodes:Conditioning (Blend), Inpainting VAE Encode (WAS), VividSharpen. Experimental nodes, or other random extra helper nodes.                                                                                                                                                                                                                                                                                                                                                                            |
| MTB Nodes                                    | [https://github.com/melMass/comfy_mtb](https://github.com/melMass/comfy_mtb)                                                           | NODES: Face Swap, Film Interpolation, Latent Lerp, Int To Number, Bounding Box, Crop, Uncrop, ImageBlur, Denoise, ImageCompare, RGV to HSV, HSV to RGB, Color Correct, Modulo, Deglaze Image, Smart Step, ...                                                                                                                                                                                                                                                                                             |
| Various ComfyUI Nodes by Type                | [https://github.com/jamesWalker55/comfyui-various](https://github.com/jamesWalker55/comfyui-various)                                   | Nodes: JWInteger, JWFloat, JWString, JWImageLoadRGB, JWImageResize, ...                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| Bmad Nodes                                   | [https://github.com/bmad4ever/comfyui_bmad_nodes](https://github.com/bmad4ever/comfyui_bmad_nodes)                                     | This custom node offers the following functionalities: API support for setting up API requests, computer vision primarily for masking or collages, and general utility to streamline workflow setup or implement essential missing features.                                                                                                                                                                                                                                                                |
| ComfyUI's ControlNet Auxiliary Preprocessors | [https://github.com/Fannovel16/comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux)                           | This is a rework of comfyui_controlnet_preprocessors based on ControlNet auxiliary models by 🤗. I think the old repo isn't good enough to maintain. All old workflow will still be work with this repo but the version option won't do anything. Almost all v1 preprocessors are replaced by v1.1 except those doesn't appear in v1.1. [w/NOTE: Please refrain from using the controlnet preprocessor alongside this installation, as it may lead to conflicts and prevent proper recognition.] |
| segment anything                             | [https://github.com/storyicon/comfyui_segment_anything](https://github.com/storyicon/comfyui_segment_anything)                         | Based on GroundingDino and SAM, use semantic strings to segment any element in an image. The comfyui version of sd-webui-segment-anything.                                                                                                                                                                                                                                                                                                                                                                 |
| Facerestore CF (Code Former)                 | [https://github.com/mav-rik/facerestore_cf](https://github.com/mav-rik/facerestore_cf)                                                 | This is a copy of [a/facerestore custom node](https://civitai.com/models/24690/comfyui-facerestore-node) with a bit of a change to support CodeFormer Fidelity parameter. These ComfyUI nodes can be used to restore faces in images similar to the face restore option in AUTOMATIC1111 webui. NOTE: To use this node, you need to download the face restoration model and face detection model from the 'Install models' menu.                                 |
| Masquerade Nodes                             | [https://github.com/BadCafeCode/masquerade-nodes-comfyui](https://github.com/BadCafeCode/masquerade-nodes-comfyui)                     | This is a node pack for ComfyUI, primarily dealing with masks.                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| Mikey Nodes                                  | [https://github.com/bash-j/mikey_nodes](https://github.com/bash-j/mikey_nodes)                                                         | Nodes: Prompt With Style, Prompt With SDXL, Resize Image for SDXL, Save Image With Prompt Data, HaldCLUT, Empty Latent Ratio Select/Custom SDXL                                                                                                                                                                                                                                                                                                                                                           |
| WAS Node Suite                               | [https://github.com/WASasquatch/was-node-suite-comfyui](https://github.com/WASasquatch/was-node-suite-comfyui)                         | A node suite for ComfyUI with many new nodes, such as image processing, text processing, and more.                                                                                                                                                                                                                                                                                                                                                                                                       |
| rgthree's ComfyUI Nodes                      | [https://github.com/rgthree/rgthree-comfy](https://github.com/rgthree/rgthree-comfy)                                                   | Nodes: Seed, Reroute, Context, Lora Loader Stack, Context Switch, Fast Muter. These custom nodes helps organize the building of complex workflows.                                                                                                                                                                                                                                                                                                                                                        |
| Use Everywhere (UE Nodes)                    | [https://github.com/chrisgoringe/cg-use-everywhere](https://github.com/chrisgoringe/cg-use-everywhere)                                 | A set of nodes that allow data to be 'broadcast' to some or all unconnected inputs. Greatly reduces link spaghetti.                                                                                                                                                                                                                                                                                                                                                                                      |
| ComfyUI Essentials                           | [https://github.com/cubiq/ComfyUI_essentials](https://github.com/cubiq/ComfyUI_essentials)                                             | Essential nodes that are weirdly missing from ComfyUI core. With few exceptions they are new features and not commodities. I hope this will be just a temporary repository until the nodes get included into ComfyUI.                                                                                                                                                                                                                                                                                     |

---

# Models

## Checkpoint

| Model Name | Download URL | File Extension |
|------------|--------------|----------------|
| DreamShaper_8_pruned.safetensors | [Download](https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors?download=true) | `.safetensors` |
| architectureinteriorsdlifechiasedammev80.safetensors | [Download](https://civitai.com/api/download/models/144610) | `.safetensors` |
| protovision-xl-high-fidelity-3d-photorealism-anime-hyperrealism-no-refiner-needed.safetensors | [Download](https://civitai.com/api/download/models/265938) | `.safetensors` |
| Realistic_Vision_V5.1.safetensors | [Download](https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/resolve/main/Realistic_Vision_V5.1.safetensors?download=true) | `.safetensors` |
| Realistic_Vision_V5.1-inpainting.safetensors | [Download](https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/resolve/main/Realistic_Vision_V5.1-inpainting.safetensors?download=true) | `.safetensors` |

---

## LoRA

| Model Name | Download URL | File Extension |
|------------|--------------|----------------|
| AnimateLCM_sd15_t2v_lora.safetensors | [Download](https://huggingface.co/wangfuyun/AnimateLCM/resolve/main/AnimateLCM_sd15_t2v_lora.safetensors?download=true) | `.safetensors` |

---

## ControlNet

| Model Name | Download URL | File Extension |
|------------|--------------|----------------|
| control_v11p_sd15_inpaint_fp16.safetensors | [Download](https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_inpaint_fp16.safetensors?download=true) | `.safetensors` |
| control_v11f1e_sd15_tile_fp16.safetensors | [Download](https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11f1e_sd15_tile_fp16.safetensors?download=true) | `.safetensors` |
| control_v11p_sd15_lineart_fp16.safetensors | [Download](https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_lineart_fp16.safetensors?download=true) | `.safetensors` |
| control_v11p_sd15_mlsd_fp16.safetensors | [Download](https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_mlsd_fp16.safetensors?download=true) | `.safetensors` |
| control_v11f1p_sd15_depth_fp16.safetensors | [Download](https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11f1p_sd15_depth_fp16.safetensors?download=true) | `.safetensors` |
| control_v11p_sd15_openpose_fp16.safetensors | [Download](https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_openpose_fp16.safetensors?download=true) | `.safetensors` |
| control_v11p_sd15_canny_fp16.safetensors | [Download](https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_canny_fp16.safetensors?download=true) | `.safetensors` |
| control_v11p_sd15_softedge_fp16.safetensors | [Download](https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_softedge_fp16.safetensors?download=true) | `.safetensors` |
| controlnet_checkpoint.ckpt (AnimateDiff ControlNet) | [Download](https://huggingface.co/crishhh/animatediff_controlnet/resolve/main/controlnet_checkpoint.ckpt?download=true) | `.ckpt` |
| control_v11e_sd15_ip2p_fp16.safetensors | [Download](https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11e_sd15_ip2p_fp16.safetensors?download=true) | `.safetensors` |

---

##  IPAdapter

| Model Name | Download URL | File Extension |
|------------|--------------|----------------|
| ip-adapter_sd15.bin | [Download](https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin) | `.bin` |
| ip-adapter_sd15_light.bin | [Download](https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_light.bin) | `.bin` |
| ip-adapter-plus_sd15.bin | [Download](https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.bin) | `.bin` |
| ip-adapter-plus-face_sd15.bin | [Download](https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus-face_sd15.bin) | `.bin` |
| ip-adapter-full-face_sd15.bin | [Download](https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-full-face_sd15.bin) | `.bin` |
| ip-adapter_sdxl.bin | [Download](https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl.bin) | `.bin` |
| ip-adapter_sdxl_vit-h.bin | [Download](https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.bin) | `.bin` |
| ip-adapter-plus_sdxl_vit-h.bin | [Download](https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin) | `.bin` |
| ip-adapter-plus-face_sdxl_vit-h.bin | [Download](https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus-face_sdxl_vit-h.bin) | `.bin` |

---

## CLIP Vision

| Model Name | Download URL | File Extension |
|------------|--------------|----------------|
| clip_vision_g.safetensors | [Download](https://huggingface.co/stabilityai/control-lora/resolve/main/revision/clip_vision_g.safetensors) | `.safetensors` |
| pytorch_model.bin (SD1.5) | [Download](https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/pytorch_model.bin) | `.bin` |
| pytorch_model.bin (SDXL) | [Download](https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/image_encoder/pytorch_model.bin) | `.bin` |

---

## Upscale

| Model Name | Download URL | File Extension |
|------------|--------------|----------------|
| 4x_NMKD-Siax_200k.pth | [Download](https://huggingface.co/gemasai/4x_NMKD-Siax_200k/resolve/main/4x_NMKD-Siax_200k.pth?download=true) | `.pth` |
| RealESRGAN_x2.pth | [Download](https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth?download=true) | `.pth` |
| RealESRGAN_x4.pth | [Download](https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth?download=true) | `.pth` |
| RealESRGAN_x8.pth | [Download](https://huggingface.co/ai-forever/Real-ESRGAN/resolve/main/RealESRGAN_x4.pth?download=true) | `.pth` |

---

## VAE

| Model Name | Download URL | File Extension |
|------------|--------------|----------------|
| vae-ft-mse-840000-ema-pruned.safetensors | [Download](https://huggingface.co/stabilityai/sd-vae-ft-mse-original/resolve/main/vae-ft-mse-840000-ema-pruned.safetensors?download=true) | `.safetensors` |

---

## Face Restore

| Model Name | Download URL | File Extension |
|------------|--------------|----------------|
| GFPGANv1.4.pth | [Download](https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth) | `.pth` |

---

## SAM Models (Segment Anything)

| Model Name | Download URL | File Extension |
|------------|--------------|----------------|
| sam_vit_h_4b8939.pth | [Download](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) | `.pth` |

---

## GroundingDino Models (Segment Anything)

| Model Name | Download URL | File Extension |
|------------|--------------|----------------|
| groundingdino_swint_ogc.pth | [Download](https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth) | `.pth` |

---
