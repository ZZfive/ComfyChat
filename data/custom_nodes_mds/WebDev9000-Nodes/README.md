# WebDev9000 Nodes

A few misc ComfyUI nodes I've created for various needs.

## IgnoreBrace

A string input node that disables "dynamic prompts", thereby allowing unescaped braces { } in your prompts.

Created to fix an issue loading LoRA with braces in their names via [ComfyUI Prompt Control](https://github.com/asagi4/comfyui-prompt-control).<br />
Without disabling dynamic prompts, braces in the filename are otherwise incompatible even when escaped.

Suggested to use with my [Extra Network Browser](https://github.com/WebDev9000/extra-network-browser/)

## SettingsSwitch

A node I made to switch between two sets of settings I use often in a high-res pass.

Plug it into Steps and Denoise on a KSampler node to quickly toggle between two presets. <br />
Low denoise = steps: 5, denoise: 0.35<br />
High denoise = steps: 10, denoise: 0.5

## Installation

Git clone this repo in the ComfyUI/custom_nodes folder.
