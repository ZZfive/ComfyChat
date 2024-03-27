# Comfy Couple

## What is

This is simple custom node for [**ComfyUI**](https://github.com/comfyanonymous/ComfyUI) which helps to generate images of actual _couples_, easier.

If you want to draw two different characters together without blending their features, so you could try to check out this custom node.

| ⭕ with Comfy Couple | ❌ without Comfy Couple  |
| --- | --- |
| ![Ayaka x Lumine](docs/images/ayalumi-comfy-couple.png) | ![Ayaka x OC](docs/images/ayalumi-plain.png) |
| _Lumine with her own hair style_ | _Lumine with hair style of Ayaka_ |

It's fork of [**laksjdjf/attention-couple-ComfyUI**](https://github.com/laksjdjf/attention-couple-ComfyUI), but implementing shortcut for the most of required nodes.

## Installation

1. Change directory to custom nodes of **ComfyUI**:

   ```bash
   cd ~/ComfyUI/custom_nodes
   ```

2. Clone this repo here:

   ```bash
   git clone https://github.com/Danand/ComfyUI-ComfyCouple.git
   ```

3. Restart **ComfyUI**.

## Usage

1. Right click in workflow.
2. Choose node: **loaders → Comfy Couple**
3. Connect inputs, connect outputs, notice **two** positive prompts for left side and right side of image respectively.

Example workflow is [here](workflows/workflow-comfy-couple.json).

## Known issues

It **is not** quite actual regional prompting.

## Comparison with [**laksjdjf/attention-couple-ComfyUI**](https://github.com/laksjdjf/attention-couple-ComfyUI)

Mask magic was replaced with comfy shortcut.

| Comfy Couple | attention-couple-ComfyUI |
| --- | --- |
| ![Comfy Couple workflow](docs/images/workflow-comfy-couple.svg) | ![attention-couple-ComfyUI workflow](docs/images/workflow-attention-couple.svg) |

## Credits

- [**@laksjdjf**](https://github.com/laksjdjf) – [original repo](https://github.com/laksjdjf/attention-couple-ComfyUI).
- [**@pythongosssss**](https://github.com/pythongosssss) – [ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts) used for capturing SVG for `README.md`
- [**@Meina**](https://civitai.com/user/Meina) – [MeinaMix V11](https://civitai.com/models/7240/meinamix) used in example.
- [**@Numeratic**](https://civitai.com/user/Numeratic) – [Genshin Impact All In One](https://civitai.com/models/108649?modelVersionId=116970) used in example.
