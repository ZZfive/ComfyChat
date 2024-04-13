# ComfyUI-HfLoader
A simple and easy to use Hugging Face model loader for ComfyUI. 

## Installation
In `custom_node` directory, run the following command to install the package:
```bash
git clone https://github.com/olduvai-jp/ComfyUI-HfLoader.git
```

And then, install the package:
```bash
pip install huggingface-hub
```

## Usage
### LoRA Loader From Hf

Load a model from Hugging Face model hub.

- `repo_name`: The name of the repository.
- `filename`: The name of the file.
- `api_token`: The API token for the Hugging Face. If repo is public, this can be left empty.
- `strength_model`: How strongly to modify the diffusion model. This value can be negative.
- `strength_clip`: How strongly to modify the CLIP model. This value can be negative.

<img width="223" alt="loraloaderfromhf" src="https://github.com/olduvai-jp/ComfyUI-HfLoader/assets/98304434/85e47571-1a02-43b4-9030-6ac58556a2c7">
