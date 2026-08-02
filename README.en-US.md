# ComfyChat
A rough LLM interpreter for ComfyUI

- [Introduction](#introduction)
- [Architecture Diagram](#architecture-diagram)
- [Deployment](#deployment)
- [Model Links](#model-links)
- [Project Development Direction](#project-development-direction)
- [Data](#data)
- [Work Content](#work-content)
  - [Data Construction](#data-construction)
  - [Training](#training)
  - [Service Deployment](#service-deployment)
- [Work Progress](#work-progress)


## Introduction
&emsp;&emsp;This project was initiated as part of the Shilong Puyu Large Model Practical Camp. Its goal is to create a rough yet comprehensive interpreter for ComfyUI using LLM capabilities, helping enthusiasts quickly learn ComfyUI and Stable Diffusion. We are deeply grateful to the Shanghai AI Laboratory for providing this opportunity, as well as to all the open-source projects and individuals who contributed to this project.

&emsp;&emsp;Having experimented with various Stable Diffusion model GUI projects out of curiosity—such as [stable diffusion webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui), [Fooocus](https://github.com/lllyasviel/Fooocus), and [ComfyUI](https://github.com/comfyanonymous/ComfyUI)—I’ve found that ComfyUI’s node-based workflow and its vibrant open-source community, with its extensive collection of custom nodes and workflows, make it a powerful tool. However, its high learning curve is a significant barrier. This project addresses this by collecting community data to build a dataset and leveraging LLM technology to create a tool that lowers the barrier to learning ComfyUI.

## Architecture Diagram
![enter image description here](assets/ComfyChat.png?raw=true)

## Deployment
&emsp;&emsp;The main project code is located in the [server](server) directory, with [demo.py](server/demo.py) serving as the entry point. Running the service is straightforward. Follow these steps:
1. It’s recommended to create a Python 3.10 virtual environment using conda or miniconda and activate it: 'conda create -n ComfyChat python=3.10' and 'conda activate ComfyChat'.
2. The [requirements.txt](server/requirements.txt) file lists the main dependencies and versions. Install them within the ComfyChat environment. Avoid using 'pip install -r requirements.txt' directly, as there are many dependencies, and a single failure will cause all installations to fail. The versions listed in requirements.txt are those used in the project development environment and are unlikely to cause conflicts, though this isn’t guaranteed.
3. Clone the project code with submodules: `git clone --recurse-submodules https://github.com/ZZfive/ComfyChat.git`.
4. For easier model management, all service-dependent models should be stored in the [weights](weights) directory. This includes models for Whisperx and GPT-SoVITS, as well as LLM and embedding models, provided they are correctly configured in [config.ini](server/config.ini). Whisperx models will be automatically downloaded to weights during demo.py execution, while GPT-SoVITS models must be manually downloaded. Create a GPT-SoVITS directory within weights and place the relevant models there. The directory structure is shown below. The weights/GPT_SoVITS/pretrained_models directory contains pre-trained models provided by the GPT-SoVITS project, available for download from the project. The remaining character voice models are community contributions and can be downloaded [here](https://www.yuque.com/baicaigongchang1145haoyuangong/ib3g1e/nwnaga50cazb2v93).
![enter image description here](assets/weights_gptsovits.png?raw=true)
5. To enable ComfyUI, prepare essential models such as Stable-diffusion, Lora, VAE, embeddings, and controlnet, and store them in the [weights](weights) directory. Modify the [visual_generations/comfyui/extra_model_paths.yaml.example](visual_generations/comfyui/extra_model_paths.yaml.example) file by removing the .example extension and renaming it to extra_model_paths.yaml. Configure the model path as shown in the screenshot below. After preparing the models, set the enable parameter to 1 in the comfyui configuration of config.ini.
![enter image description here](assets/comfyui_models.png?raw=true)
6. After completing these steps and correctly configuring [config.ini](server/config.ini), run the service with `python demo.py`.

## Model Links
- [zzfive/ComfyChat-Llama3-8b-instruct-v2-2](https://huggingface.co/zzfive/ComfyChat-Llama3-8b-instruct-v2-2)
- [zzfive/ComfyChat-Qwen2-7b-instruct-v2-2](https://huggingface.co/zzfive/ComfyChat-Qwen2-7b-instruct-v2-2)
- [zzfive/ComfyChat-InternLM2.5-7b-v2-2](https://huggingface.co/zzfive/ComfyChat-InternLM2.5-7b-v2-2)
- [zzfive/ComfyChat-InternLM2.5-7b-v2-1](https://huggingface.co/zzfive/ComfyChat-InternLM2.5-7b-v2-1)
- [zzfive/ComfyChat-InternLM2-7b-v2-1](https://huggingface.co/zzfive/ComfyChat-InternLM2-7b-v2-1)
- [zzfive/ComfyChat-InternLM2-1-8b-v2-1](https://huggingface.co/zzfive/ComfyChat-InternLM2-1-8b-v2-1)

## Project Development Direction
- Completed:
   - [x] Fine-tuning: Underwent fine-tuning for LLMs like InternLM2 and LLaMA3.
   - [x] RAG: Developed a ComfyUI knowledge base using open-source RAG framework to enhance LLM response accuracy.
   - [x] Multimodal Integration: Integrated functionalities such as ASR, TTS, and image generation.
- Pending:
   - [ ] Workflow Training: Since ComfyUI workflows are essentially JSON documents representing node connections, similar to code, we aim to explore whether LLMs can understand and construct workflows through training and testing.

## Data

Data Sources
 - Explanation of Default Nodes: Collected from ComfyUI project documentation, online resources, and community node documentation:
   - [https://github.com/BlenderNeko/ComfyUI-docs](https://github.com/BlenderNeko/ComfyUI-docs)
 - Explanation of Custom Nodes: Collected from various custom node project documentation:
   - Batch download of custom node project list files from [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager).
   - [https://github.com/6174/comflowy](https://github.com/6174/comflowy)
   - [https://github.com/get-salt-AI/SaltAI-Web-Docs](https://github.com/get-salt-AI/SaltAI-Web-Docs)
   - [https://github.com/CavinHuang/comfyui-nodes-docs](https://github.com/CavinHuang/comfyui-nodes-docs)
 - Workflow Files: Not yet actively pursued.
   - [c station](https://civitai.com/)
   - Personal summaries on GitHub.
   - [https://comfyworkflows.com](https://comfyworkflows.com)
   - [https://openart.ai/workflows](https://openart.ai/workflows)

## Work Content

### Data Construction
 - Data Collection:
   - ComfyUI Description Data Collection:
     - [x] Collected community documentation information [here](https://blenderneko.github.io/ComfyUI-docs/#further-support).
     - [x] Collected node list information from ComfyUI Manager.
     - [ ] Internet-based ComfyUI information collection (in progress).
     - [x] Deployed video and audio-to-text model for text collection (Whisperx).
   - ComfyUI Workflow and Text Description Data Collection:
     - [ ] Pending.
   - Deduplication and Security Filtering?

 - Dataset Construction:
   - Built dialogue data for instruction fine-tuning: v1 and v2 datasets are available [here](data/message_jsons/README.md).

 ### Training
 - [ ] LLaMA3 Fine-Tuning
 - [x] InternLM2.5-7b Fine-Tuning
 - [x] InternLM2-7b Fine-Tuning
 - [x] InternLM2-1.8b Fine-Tuning

 ### Service Deployment
 - [x] Deployed dialogue service based on fine-tuned LLMs.
 - [x] Built RAG service.
 - [x] Integrated service with TTS.
 - [x] Integrated service with image generation.

## Work Progress
 - [ ] Dataset Optimization:
   - [ ] Collected documents or extracted data from videos for construction.
   - [x] Updated custom node data based on ComfyUI-Manager, combined with community data to create v2_2 dataset [v2_2 dataset](data/message_jsons/v2/comfyui_data_v2_2.json).
 - [x] External Interface Development:
   - [x] Developed inference module (based on Gradio).
   - [x] Developed TTS module (based on GPT-SoVITS).
   - [x] Developed ASR module (based on WhisperX).
   - [x] Integrated service with image generation (based on ComfyUI).
 - [x] RAG System Development:
   - [x] System setup.
   - [x] Converted constructed data into vector storage.
 - [x] Model Training Based on v2 Dataset:
   - [x] Constructed data for four community-focused node text projects.
 - [x] Fine-tuned InternLM2-chat-1.8b and InternLM2-chat-7b based on v1 dataset; models are [zzfive/ComfyChat-InternLM2-1-8b-v1](https://huggingface.co/zzfive/ComfyChat-InternLM2-1-8b-v1) and [zzfive/ComfyChat-InternLM2-7b-v1](https://huggingface.co/zzfive/ComfyChat-InternLM2-7b-v1).
 - [x] Constructed Chinese and English fine-tuning datasets using DeepSeek, Kimi, and other LLMs, combined with Aplaca-GPT4 dataset to create v1 dataset.
 - [x] Collected a batch of ComfyUI and custom node project documentation (readme and description text).


## Acknowledgements
We extend our heartfelt thanks to the following teams, projects, and individuals:
 - Shanghai AI Laboratory and Shilong Puyu Team.
 - [ComfyUI](https://github.com/comfyanonymous/ComfyUI).
 - [https://github.com/BlenderNeko/ComfyUI-docs](https://github.com/BlenderNeko/ComfyUI-docs).
 - [https://github.com/6174/comflowy](https://github.com/6174/comflowy).
 - [https://github.com/get-salt-AI/SaltAI-Web-Docs](https://github.com/get-salt-AI/SaltAI-Web-Docs).
 - [https://github.com/CavinHuang/comfyui-nodes-docs](https://github.com/CavinHuang/comfyui-nodes-docs).
 - [HuixiangDou](https://github.com/InternLM/HuixiangDou).
 - [whisperX](https://github.com/m-bain/whisperX).
 - [whispercpp](https://github.com/AIWintermuteAI/whispercpp).
 - [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS).
 - [Chattts](https://github.com/2noise/ChatTTS).
 - [旭_1994](https://blog.csdn.net/qq_38944169?type=blog)'s [CSDN Blog](https://blog.csdn.net/qq_38944169/article/details/139245317).
