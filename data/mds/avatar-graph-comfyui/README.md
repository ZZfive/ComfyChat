# avatar-graph-comfyui

![image](https://avatech-avatar-dev1.nyc3.cdn.digitaloceanspaces.com/public-download/github-readme/270574817-201a005b-7e00-4671-85a1-54937bf0704e.png)

Wanna animate or got a question? Join our [Discord](https://discord.gg/Xp6mZ4Ez5P)

A custom nodes module for **creating real-time interactive avatars** powered by blender bpy mesh api + Avatech Shape Flow runtime.

> **WARNING**  
> We are still making changes to the nodes and demo templates, please stay tuned.

# Demo

| <img src="https://github.com/avatechai/avatar-graph-comfyui/assets/73209427/12e2bfc6-438e-4d16-bead-9957ced3bae1" width="180"/><br>[Interact 👆](https://editor.avatech.ai/viewer?avatarId=cce15b92-6d1c-4966-91b9-362d7833cb5d) | <img src="https://github.com/avatechai/avatar-graph-comfyui/assets/73209427/0c497025-7ed5-4e25-b4d1-5a257e1ba814" width="180"/><br>[Interact 👆](https://editor.avatech.ai/viewer?avatarId=42a8182f-b140-48c0-a556-35cddf0f76f7) | <img src="https://github.com/avatechai/avatar-graph-comfyui/assets/73209427/a2bf71e3-0d9c-4ddd-957f-a6b0cb7e622a" width="180"/><br>[Interact 👆](https://editor.avatech.ai/viewer?avatarId=7c23b8d6-d1a5-41c7-a084-250461dbef22) | <img src="https://github.com/avatechai/avatar-graph-comfyui/assets/73209427/ad808c42-5297-4e61-8be8-d5cb7729d2ff" width="180"/><br>[Interact 👆](https://editor.avatech.ai/viewer?avatarId=268b32c4-f9b9-4db8-a27c-a7e974f0f0ac) |
| :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| <img src="https://github.com/avatechai/avatar-graph-comfyui/assets/73209427/1d1ad8f9-31a6-48ec-bad2-ce972ee3b12f" width="180"/><br>[Interact 👆](https://editor.avatech.ai/viewer?avatarId=f97fc5bb-93b0-4b02-bbc0-327dd41d0fc5) | <img src="https://github.com/avatechai/avatar-graph-comfyui/assets/73209427/06958585-f780-4b38-8f5d-bddabd7da78a" width="180"/><br>[Interact 👆](https://editor.avatech.ai/viewer?avatarId=4d50aa03-26e4-47e7-97b6-c3fe9d8fc96e) | <img src="https://github.com/avatechai/avatar-graph-comfyui/assets/73209427/3d0e6b54-d45f-45ac-90bd-d8b149880f98" width="180"/><br>[Interact 👆](https://editor.avatech.ai/viewer?avatarId=791014cb-7836-4641-afdb-ac331064b682) | <img src="https://github.com/avatechai/avatar-graph-comfyui/assets/73209427/1d1ad8f9-31a6-48ec-bad2-ce972ee3b12f" width="180"/><br>[Interact 👆](https://editor.avatech.ai/viewer?avatarId=f97fc5bb-93b0-4b02-bbc0-327dd41d0fc5) |

# How to?

- [Basic Rigging Workflow Template](#basic-rigging-workflow-template)
- [Best Practices for image input](#best-practices-for-image-input)
- [Custom Nodes List](#custom-nodes)
- [Shape Flow](#shape-flow)
- [Installation](#installation)
- [Development](#development)
- [Join Discord 💬](https://discord.gg/WNtBYksDwF)

# Basic Rigging Workflow Template

### 1. Creating an eye blink and lipsync avatar

![ComfyUI_00668_](https://github.com/avatechai/avatar-graph-comfyui/assets/73209427/72a0abe8-2482-4a0d-8436-8eb231fd2f6d)

For optimal results, please input a character image with an open mouth and a minimum resolution of 768x768. This higher resolution will enable the tool to accurately recognize and work with facial features.

Download: Save the image, and drag into Comfyui or [Simple Shape Flow](https://github.com/avatechai/avatar-graph-comfyui/blob/main/workflow_templates/SimpleEye+MouthMovement.json)

### 2. Creating an eye blink and lipsync emoji avatar

| ![ComfyUI_00045_](https://github.com/avatechai/avatar-graph-comfyui/assets/18395202/b4787166-85df-43c6-9fe9-252989f68d18) | ![emoji_480p30_high](https://github.com/avatechai/avatar-graph-comfyui/assets/18395202/7d8b2b0a-e979-421d-8055-b4acac50a0c1) | 
| :--: | :--: |

Download: Save the image, and drag into Comfyui

| ![ComfyUI_09609_](https://github.com/avatechai/avatar-graph-comfyui/assets/73209427/caa98eec-4fb2-449d-9558-5d4a45e07580) | ![dog_480p15_high](https://github.com/avatechai/avatar-graph-comfyui/assets/73209427/9fb701c9-f25c-408f-b96c-749773a53bd2) | 
| :--: | :--: |

Download: Save the image, and drag into Comfyui or [Dog Workflow](https://github.com/avatechai/avatar-graph-comfyui/blob/main/workflow_templates/Dog_workflow.json)

# Best practices for image input

### 1. Generate a new character image

![image](https://avatech-avatar-dev1.nyc3.cdn.digitaloceanspaces.com/public-download/github-readme/270609114-acea9933-359b-4398-8d2a-582bf02bef99.png)

We need a character image with an open mouth and enable the tool to easily recognize facial features, so please add to the prompt:
`looking at viewer, detailed face, open mouth, [smile], solo,eye-level angle`

Download: [Character Gen Template](https://github.com/avatechai/avatar-graph-comfyui/blob/main/workflow_templates/SimpleCharacterGen.json)

### 2. Make existing character image mouth open (Inpaint)

![inpaint_workflow](https://avatech-avatar-dev1.nyc3.cdn.digitaloceanspaces.com/public-download/github-readme/270589181-d11d840b-7ea6-4b47-bc26-a2af7c8c27a5.png)

To maintain consistency with the base image, it is recommended to utilize a checkpoint model that aligns with its style.

Download: [Mouth Open Inpaint Template](<https://github.com/avatechai/avatar-graph-comfyui/blob/main/workflow_templates/MouthOpen_(inpaint).json>)

<details>
<summary> Inpaint Demonstration </summary>

<video src="https://github.com/avatechai/avatar-graph-comfyui/assets/73209427/e3b77295-a1bf-4d96-9551-7cc423a4af73"/>

</details>

### 3. Pose Constraints (ControlNet)

![image](https://avatech-avatar-dev1.nyc3.cdn.digitaloceanspaces.com/public-download/github-readme/270943267-c3cae113-2df4-45f2-a19c-885cbee75450.png)

Place normal and openpose image with reference to images.

Download: [ControlNet Gen](https://github.com/avatechai/avatar-graph-comfyui/tree/main/workflow_templates/TemplateGen01)


# Recommend Checkpoint Model List

##### Anime Style SD1.5

- https://civitai.com/models/35960/flat-2d-animerge
- https://civitai.com/models/24149/mistoonanime

##### Realistic Style SD1.5

- https://civitai.com/models/4201/realistic-vision-v51
- https://civitai.com/models/49463/am-i-real
- https://civitai.com/models/43331/majicmix-realistic

# Custom Nodes

Expand to see all the available nodes description.

<details>
<summary> All Custom Nodes </summary>

| Name                 | Description                                                                                    | Preview                                                                                                                                                              |
| -------------------- | ---------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Segmentation (SAM)` | Integrative SAM node allowing you to directly select and create multiple image segment output. | <img src="https://avatech-avatar-dev1.nyc3.cdn.digitaloceanspaces.com/public-download/github-readme/270576351-8aabeba8-5450-4d39-8203-e91f9ab47190.png" width="300"> |


| Name                         | Description                                                                                     | Preview                                                                                                                                                              |
| ---------------------------- | ----------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Create Mesh Layer`          | Create a mesh object from the input images (usually a segmented part of the entire image)       | <img src="https://avatech-avatar-dev1.nyc3.cdn.digitaloceanspaces.com/public-download/github-readme/270576646-40740d25-9411-4cd3-a6c0-8b9008bca41c.png" width="300"> |
| `Join Meshes`                | Combine multiple meshes into a single mesh object                                               | <img src="https://avatech-avatar-dev1.nyc3.cdn.digitaloceanspaces.com/public-download/github-readme/270577004-ba7afbc5-9cd5-4f97-9614-f71133f5783e.png" width="300"> |
| `Match Texture Aspect Ratio` | Since the mesh is created in 1:1 aspect ratio, a re-scale is needed at the end of the operation | <img src="https://avatech-avatar-dev1.nyc3.cdn.digitaloceanspaces.com/public-download/github-readme/270602782-cb7155be-fb31-49f8-a24a-d001a1484ea7.png" width="300"> |
| `Plane Texture Unwrap`       | Will perform mesh face fill and UV Cube project on the target plane mesh, scaled to bounds.     | <img src="https://avatech-avatar-dev1.nyc3.cdn.digitaloceanspaces.com/public-download/github-readme/270603006-4b9c0cf5-0497-47bf-8e06-5a3370084c11.png" width="300"> |

| Name                    | Description                                                                                        | Preview                                                                                                                                                              |
| ----------------------- | -------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Mesh Modify Shape Key` | Given shape key name & target vertex_group, modify the vertex / all vertex’s transform             | <img src="https://avatech-avatar-dev1.nyc3.cdn.digitaloceanspaces.com/public-download/github-readme/270577944-ab4f259c-89a7-4f51-bc54-fd179e252073.png" width="300"> |
| `Create Shape Flow`     | Create runtime shape flow graph, allowing interactive inputs affecting shape keys value in runtime | <img src="https://avatech-avatar-dev1.nyc3.cdn.digitaloceanspaces.com/public-download/github-readme/270592752-abfdd801-0387-4c5d-9c11-6c23337ff1dd.png" width="300"> |

| Name                 | Description                                                                                           | Preview                                                                                                                                                              |
| -------------------- | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Avatar Main Output` | The primary output of the .ava file. The embedded Avatar View will auto update with this node's output | <img src="https://avatech-avatar-dev1.nyc3.cdn.digitaloceanspaces.com/public-download/github-readme/270592519-6a9a8bb4-05ec-4a2e-98bf-194b6af3a62a.png" width="300"> |

</details>

# Shape Flow

![image](https://avatech-avatar-dev1.nyc3.cdn.digitaloceanspaces.com/public-download/github-readme/270618471-a834e535-4f87-4b77-81a6-435e3a67ca4a.png)

# Installation

## Method 1 - Windows

1. Download Python environment from [here](https://avatech-avatar-dev1.nyc3.digitaloceanspaces.com/comfyui/ComfyUI_3.10.7z)

2. Unzip it to ComfyUI directory

3. Run the `run_cpu_3.10.bat` or `run_nvidia_gpu_3.10.bat`

4. Install avatar-graph-comfyui from [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)

## Method 2 - macOS/Linux

Make sure your Python environment is `3.10.x` as required by the [bpy](https://pypi.org/project/bpy/) package. Then go to the [ComfyUI](https://github.com/comfyanonymous/ComfyUI) directory and run:

> Suggest using conda for your comfyui python environment
> 
> `conda create --name comfyui python=3.10`
> 
> `conda activate comfyui`
> 
> `pip install -r requirements.txt`

1. `cd custom_nodes`

2. `git clone https://github.com/avatechgg/avatar-graph-comfyui.git`

3. `cd avatar-graph-comfyui && python -m pip install -r requirements.txt`

4. Restart ComfyUI with enable-cors-header `python main.py --enable-cors-header` or (for mac) `python main.py --force-fp16 --enable-cors-header`

# Development

If you are interested in contributing

For comfyui frontend extension, frontend js located at `avatar-graph-comfyui/js`

Web stack used: [vanjs](https://github.com/vanjs-org/van) [tailwindcss](https://github.com/tailwindlabs/tailwindcss)

## Install deps

```
pnpm i
```

Run the dev command to start the tailwindcss watcher

```
pnpm dev
```

For each changes, simply refresh the comfyui page to see the changes.

<details>
<summary>p.s. For tailwind autocomplete, add the following to your vscode settings.json.</summary>
    
```json
{
    "tailwindCSS.experimental.classRegex": [
        ["class\\s?:\\s?([\\s\\S]*)", "(?:\"|')([^\"']*)(?:\"|')"]
    ]
}
```
</details>

## Update blender node types

To update blender operations input and output types (stored in `blender/input_types.txt`), run:

```bash
python generate_blender_types.py
```

# FAQ

## What is `--enable-cors-header` used for?
It is used to enable communication between ComfyUI and our editor (https://editor.avatech.ai), which is in charge of animating static characters. The only messages exchanged between them are the character data like the meshes of eyes and mouth, and the JSON format of our editor graph.

When you execute the ComfyUI graph, it sends the character data and the JSON graph to our editor for animating. When you modify and save the graph in our editor, it sends the modified graph back to ComfyUI. To validate it, you can open the `js/index.js`, and log the message in `window.addEventListener("message", ...)` and `postMessage(message)`.

You can also run ComfyUI *without* the `--enable-cors-header`: execute the ComfyUI workflow, then download the .GLB or .GLTF format by right clicking the Avatar Main Output node and Save File option. Yet, this will disable the real-time character preview in the top-right corner of ComfyUI. Feel free to view it in other software like Blender.
