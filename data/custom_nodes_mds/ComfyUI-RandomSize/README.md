# ComfyUI-RandomSize

A ComfyUI custom node that randomly selects a height and width pair from a list in a config file. 
The core use is to provide variety in generations of large numbers of images at once.

Presets included:
* 9 SD 1.5-friendly resolutions.
* 9 SDXL-friendly resulotions
* 9 Resolutions centered around 512x512
* 9 around 640x640
* 13 around 768x768
* 15 around 896x896
* 17 around 1024x1024

## Install

```
cd [path to ComfyUI]/custom_nodes
git clone https://github.com/JerryOrbachJr/ComfyUI-RandomSize.git
```

## Usage
* Select a seed. The same seed should produce the same size every time
* Control After(/Before) Generate: set to increment, decrement, or random to get a random size from the preset each generation.
* Select a preset. The default preset is SD 1.5 if none is chosen.
* Run your queue.

### Adding Presets

- Add a text file with name of your choosing and extension ".yaml" in [path to ComfyUI]/custom_nodes/ComfyUI-RandomSize/sizes/custom
- The first line of your file needs to be ```sizes:```
- Every successive line should be a size in the format ```- WIDTHxHEIGHT``` replacing WIDTH and HEIGHT with pixel numbers e.g. ```- 768x512```
- Restart Comfy to see your Preset in the list
