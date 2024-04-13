# Xycuno Oobabooga

> :warning: These custom nodes are under initial development and may be
> substantially changed between updates.

Xycuno Oobabooga provides custom nodes for
[ComfyUI](https://github.com/comfyanonymous/ComfyUI), for sending requests to
an [Oobabooga](https://github.com/oobabooga/text-generation-webui) instance to
assist in creating prompt texts.

## Installation

### git clone

1. `cd` to the `custom_nodes` directory of your ComfyUI installation
2. `git clone https://github.com/Xyem/Xycuno-Oobabooga.git`
3. Re/start ComfyUI

### git submodule

If you maintain your installation of ComfyUI as a git repository, you can use
submodules for custom nodes. This is the node author's preference of
installation/management.

1. `cd` to the `custom_nodes` directory of your ComfyUI installation
2. `git submodule add -f https://github.com/Xyem/Xycuno-Oobabooga.git xycuno_oobabooga`
3. `git add ../.gitmodules`
4. `git add xycuno_oobabooga`
5. `git commit -m "Add Xycuno Oobabooga custom nodes"`

This can then be updated:
 
1. `cd` to the `custom_nodes` directory of your ComfyUI installation
2. `git submodule update --remote xycuno_oobabooga`
3. `git add ../.gitmodules`
4. `git commit -m "Update Xycuno Oobabooga custom nodes"`

## Nodes

### Oobabooga

This node is for having Oobabooga complete the rest of the provided text.

#### Inputs

##### control_after_generate

Sets the behaviour of changes to `seed`, if any, when the workflow has been
generated. Automatically added by ComfyUI itself, defaults to `randomize`.

##### host

Address of the Oobabooga API, defaults to `http://127.0.0.1:5000`.

##### max_tokens

Maximum number of tokens to be used to complete the prompt. Defaults to `200`.

##### prompt

The text to be completed. Defaults to an instruction to describe an image
featuring the ComfyUI mascot.

##### seed

Seed to be used. Defaults to `0`.

##### temperature

Defaults to `1.0`.

##### top_p

Defaults to `0.9`.

#### Outputs

##### response

The text returned by Oobabooga, not including `prompt`.
