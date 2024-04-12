# NX_PromptStyler

A custom node for ComfyUI to create a prompt based on a list of keywords saved in CSV files.

<img src="https://img.shields.io/badge/ComfyUI-green" /> <img src="https://img.shields.io/badge/Python-3.10-blue" /> <img src="https://img.shields.io/badge/Custom Node-1.0.0.Stable-orange" /> [![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](http://perso.crans.org/besson/LICENSE.html)

This node was inspired by PCMonsterx's [ComfyUI-CSV-Loader](https://github.com/PCMonsterx/ComfyUI-CSV-Loader) and by modifications and additions made by [Adel AI](https://www.facebook.com/AI.ArtByAdel). **Thanks to them!**

> [!IMPORTANT]
> Node tested only on Linux. It should work without any problems on Windows and Mac but has not been tested on these OS.

## Installation

If GIT is installed on your system, go to the `custom_nodes` subfolder of your ComfyUI installation, open a terminal and type: 
```:bash
git clone https://github.com/Franck-Demongin/NX_PromptStyler.git
```

If GIT is not installed, retrieve the [ZIP](https://github.com/Franck-Demongin/NX_PromptStyler/archive/refs/heads/main.zip) file, unzip it into the `custom nodes` folder and rename it NX_PromptStyler.

Restart ComfyUI. ***NX_PromptStyler*** should be available in the ***NX_Nodes*** category.

## Note about CSV

The node detects CSV files placed in the _CSV_ subfolder. 
Copy files from CSV_default to CSV folder. They are automatically taken into account and classified in a predetermined order: 
1. _style.csv_
2. _framing.csv_
3. _cameras.csv_
4. _lighting.csv_
5. _effects.csv_
6. _composition.csv_
7. _films.csv_
8. _artists.csv_

You can add other files and they will appear next, listed in alphabetical order.

The _positive.csv_ and _negative.csv_ files will appear at the end of the list.

The CSV files used must comply with the following characteristics:
- be encoded in UTF-8
- include a header line consisting of name,prompt,negative_prompt (negative prompt are not used for the moment!)
- use a comma as a separator and " as a delimiter 

## Features

- select or ignore a category (leave the value at "_None_" to ignore it)
- assign a positive or negative weight to each category
- preset management, creating, editing and deleting favourite settings. 
- display of positive and negative prompts generated by the node (see below)

## About generate options

It may be useful to display the prompts resulting from the use of the node before starting image generation. This allows you to check the content of the generated prompts and allows you to modify them before they are sent to the workflow.

If you use this option, bear in mind that, as a last resort, the contents of these fields (output positive and output negative) will be used as the node's output.

If you want to use other settings (categories or presets), empty the output fields or regenerate the prompts before launching the workflow! 