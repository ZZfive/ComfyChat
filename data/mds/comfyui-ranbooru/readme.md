# Ranbooru for ComfyUI
![Alt text](assets/pics/ranbooru.png)
Ranbooru is an extension for the [comfyUI](https://github.com/comfyanonymous/ComfyUI). The purpose of this extension is to add a node that gets a random set of tags from boorus pictures. This is mostly being used to help me test my checkpoints on a large variety of tags.
![Alt text](assets/pics/image.png)

## Installation
Just clone this repository into the custom_nodes folder of ComfyUI. Restart ComfyUI and the extension should be loaded.

## Features
These are the nodes available in the Ranbooru extension:
### Ranbooru  
This node will get a random set of tags from boorus pictures.
Parameters:
- `Booru`: The booru to get the tags from.
- `Tags`: The tags to search for.
- ~~`remove_tags`: The tags to remove from the search.~~ replaced as a new node
- ~~`max_tags`: The maximum amount of tags to get.~~ replaced as a new node
- `rating`: The mature rating of the picture.
- `change_color`: Change this if you want a colored or a black and white picture.
- `use_last_prompt`: If you want to use the last prompt as the tags.
- `return_picture`: If you want to return the picture as well.

### Ranbooru URL
This node will get the tags from a specific picture.
- `booru`: The booru to get the tags from.
- `url`: The url of the picture. You can also pass the ID of the picture.
- `return_picture`: If you want to return the picture as well.

### Random Picture Path
This node will get a random picture from a specific path.
- `path`: The path to get the picture from.
- `include_subfolders`: If you want to include subfolders in the search.

### PromptMix

The `PromptMix` node is used to mix the words in a given prompt. It supports three types of mixing: 'Shuffle', 'Reverse', and 'Inverse'. The type of mix is determined by the `mix_type` parameter.
- `prompt`: A string that represents the prompt to be mixed.
- `delimiter`: A string that represents the delimiter used to split the prompt into words.
- `mix_type`: A string that represents the type of mix to be applied. It can be 'Shuffle', 'Reverse', or 'Inverse'.

### PromptLimit

The `PromptLimit` node is used to limit the number of words in a given prompt. The number of words is determined by the `limit` parameter.
- `prompt`: A string that represents the prompt to be limited.
- `separator`: A string that represents the separator used to split the prompt into words.
- `limit`: An integer that represents the maximum number of words allowed in the prompt.
 
### PromptRandomWeight
The `PromptRandomWeight` function is used to randomly select one or more words from a given prompt and give them a random weight based on a user-defined range. The number of words to be selected is determined by the `num_words` parameter.
- `prompt`: A string that represents the prompt to be weighted.
- `separator`: A string that represents the separator used to split the prompt into words.
- `min_weight_value`: A float that represents the minimum weight value allowed.
- `max_weight_value`: A float that represents the maximum weight value allowed.
- `max_weight_tags`: An integer that represents the number of words to be selected.
- `order`: A string that represents the order in which the words will be selected. It can be 'random' or 'ordered'.

### PromptBackground
The `PromptBackground` function is used to add a background context to a given prompt. 
- `prompt`: A string that represents the prompt to be given a background context.
- `background`: A string that represents the background context to be added to the prompt.

### PromptRemove
The `PromptRemove` function is used to remove specified words from a given prompt.

- `prompt`: A string that represents the prompt to be processed.
- `words_to_remove`: A list of strings that represents the words to be removed from the prompt.

### LockSeed
The `LockSeed` node is used to lock the seed for the random number generator. This is useful when you want to generate the same random numbers across different nodes. 

- `use_last`: Lock the seed to the last seed used.

### TimestampFileName
The `TimestampFileName` node is used to generate a timestamp-based file name. The file name is generated based on the current date and time.

- `filename`: A string that represents the prefix to be added to the file name.

## Found an issue?  
If you found an issue with the extension, please report it in the issues section of this repository.  

## Check out my other scripts for 1111automatic
- [Ranbooru](https://github.com/Inzaniak/sd-webui-ranbooru)
- [Workflow](https://github.com/Inzaniak/sd-webui-workflow)

---
## Made by Inzaniak
![Alt text](assets/pics/logo.png) 


If you'd like to support my work feel free to check out my Patreon: https://www.patreon.com/Inzaniak

Also check my other links:
- `Personal Website`: https://inzaniak.github.io 
- `Deviant Art`: https://www.deviantart.com/inzaniak
- `CivitAI`: https://civitai.com/user/Inzaniak/models
