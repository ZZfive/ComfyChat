# Llama for ComfyUI

# Description

## What This Is
Llama for ComfyUI is a tool that allows you to run language learning models (LLMs) within ComfyUI.  Sort of a glue to take a cool AI tool
from one place and let us use it somewhere else.

## Where This Fits
- LLMs are files that store an AI that can read and write text.
- [llama-cpp](https://github.com/ggerganov/llama.cpp) is a command line program that lets us use LLMs that are stored in the GGUF file format from [huggingface.co](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending&search=GGUF)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) lets us use llama.cpp in Python.
- [stable diffusion](https://huggingface.co/spaces/stabilityai/stable-diffusion) is a command line program that lets us use image generation AI models.
- [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) lets us use Stable Diffusion using a flow graph layout.
- So ComfyUI-Llama (that's us!)lets us use LLMs in ComfyUI.

## Why I Made This
- I wanted to integrate text generation and image generation AI in one interface and see what other people can come up with to use them.

## Features:
- Currently let's you easily load GGUF models in a consistent fashion with other ComfyUI models and can use them to generate strings of output text with seemingly correct seeding and temperature.

- Works well with [ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts) and using the [ShowText](https://github.com/pythongosssss/ComfyUI-Custom-Scripts#show-text) Node to get output from the LLM.

## Upcoming Features:
- Intend to discover how to improve interactivity so you can get a dialogue going

# Installation

## What you need first:
- [Python](https://github.com/python)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## Highly Recommended
- [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)
- [ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts)

## Steps if using Comfy Manager:
1. Visit your Install Custom Nodes page, and search for ComfyUI-Llama.
2. Hit Install and restart when prompted.
3. Copy your GGUF files into ```./ComfyUI/custom_nodes/ComfyUI-Llama/models/*```
4. Hit Ctrl+F5 to hard reload the browser window.
5. The nodes should be in the LLM menu.

## Steps if installing manually:
1. Clone this repo into `custom_nodes` folder.
2. Install [llama-cpp-python](https://github.com/abetlen/llama-cpp-python).
3. Copy your GGUF files into ```./ComfyUI/custom_nodes/ComfyUI-Llama/models/*```
4. Hit Ctrl+F5 to hard reload the browser window.
5. The nodes should be in the LLM menu.

## If you can't install:
Either post an issue on github, or ask [on Element in Comfy's channel](https://matrix.to/#/#comfyui:matrix.org)

# Usage

## Instructions:
1. Download GGUF language learning models, which can be found on HuggingFace. You will need at least 1. Different models produce different results.

2. Place models in ```ComfyUI/custom_nodes/ComfyUI-Llama/models```. They can be renamed if you want.

3. Fire up/Restart ComfyUI and allow it to finish restarting.

4. Hit Ctrl+F5 to ensure the browser is refreshed.

5. Check your ComfyUI available nodes and find the LLM menu.

6. Load LLM Model Basic

This is a simplified call of this:
[llama-cpp-python's init method](https://abetlen.github.io/llama-cpp-python/#llama_cpp.llama.Llama.__init__)

7. Call LLM Model Basic

This is a simplified call of this:

[llama-cpp-python's call method](https://abetlen.github.io/llama-cpp-python/#llama_cpp.llama.Llama.__call__)

## If you get errors:
Either post an issue on github, or ask [on Element in Comfy's channel](https://matrix.to/#/#comfyui:matrix.org)

## Examples

Example 1 shows the two most basic nodes in their simplest setup.  The call node will output the results as a string.
You would have to use a custom node to show text.  If you don't have one, I would suggest using ComfyUI-Custom-Script's ShowText node.

![image](https://github.com/daniel-lewis-ab/ComfyUI-Llama/blob/main/example1.png)

Example 2 shows a slightly more advanced configuration that suggests changes to human written python code.  This is
just one of several workflow tools that I have at my disposal.

![image](https://github.com/daniel-lewis-ab/ComfyUI-Llama/blob/main/example2.png)

See the documentation for [llama-cpp-python](https://abetlen.github.io/llama-cpp-python/) on that interface

# For Possible Contributors

## Known Issues
- Load Model Advanced's kwargs, and chathandler are disabled
- Call Advanced's stream, logprobs, stop, stopping_criteria, logits_processor, logits_bias, grammar are disabled
- Create Completion's stream, logprobs, stop, stopping_criteria, logits_processor, logits_bias, grammar are disabled
- Create Embedding method is disabled
- Generate is disabled
- Create Chat Completion is disabled
- Llava has not been implemented yet

- No known way to loopback output from an LLM model repeatedly.

    This may be resolved by using [ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts)
    and using the [Repeater](https://github.com/pythongosssss/ComfyUI-Custom-Scripts#wip-repeater) Node, but it's
    as of yet a WIP.

- Haven't widely tested strange abuses of inputs.

- Error handling and logging is still not mature.

# Conclusion

We appreciate your interest in Llama for ComfyUI. Feel free to explore and provide feedback or report any issues you encounter. Your contributions and suggestions are valuable to the project.




