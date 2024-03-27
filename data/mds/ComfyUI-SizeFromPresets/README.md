# ComfyUI-SizeFromPresets
![SizeFromPresets Preview](preview.png "SizeFromPresets Preview")  
日本語版READMEは[こちら](README.jp.md)。

- Custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).
- Add nodes that outputs width and height of the size selected from the preset.

## Installation
```
cd <ComfyUI directory>/custom_nodes
git clone https://github.com/nkchocoai/ComfyUI-SizeFromPresets.git
```

## Nodes
### Size From Presets (SD1.5)
- Select image size presets for SD1.5.
- Output width and height of the selected preset.
- Presets can be set from [presets/sd15.csv](presets/sd15.csv).

### Size From Presets (SDXL)
- Select image size presets for SDXL.
- Output width and height of the selected preset.
- Presets can be set from [presets/sdxl.csv](presets/sdxl.csv).

### Empty Latent Image From Presets (SD1.5)
- Select image size presets for SD1.5.
- Output width and height of the selected preset.
- Presets can be set from [presets/sd15.csv](presets/sd15.csv).

### Empty Latent Image From Presets (SDXL)
- Select image size presets for SDXL.
- Output empty latent image, width and height of the selected preset.
- Presets can be set from [presets/sdxl.csv](presets/sdxl.csv).

### Random ... From Presets (SD...)
- Select randomly from the presets listed in the CSV file.
- In the case of Size, output width and height of the selected preset.
- In the case of Empty Latent Image, output empty latent image, width and height of the selected preset.