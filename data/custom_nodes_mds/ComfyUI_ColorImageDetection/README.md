# ComfyUI Color Detection Nodes
A collection of nodes for detecting color in images, leveraging RGB and LAB color spaces. These nodes aim to distinguish colored images from black and white, including those with color tints.

# Installation

- Git clone the repo into `ComfyUI/custom_nodes`.
- Or, install with [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager).
- Alternatively, download the `.py` files into your `ComfyUI/custom_nodes` folder. Ensure `requirements.txt` is installed for full functionality.

## Limitations

Both RGB and LAB color detection methods have challenges in reliably differentiating between color-tinted black and white pictures and fully colored images. RGB Color Detection tends to be more reliable for detecting color tints but operates slower. In contrast, LAB Color Detection offers faster processing times but may not be as effective for nuanced tint detection.

## Features

### RGB Color Detection
Analyzes RGB color space deviations to determine image coloration. Offers customizable pixel percentage consideration for enhanced accuracy, especially useful for tinted images.

### LAB Color Detection
Utilizes LAB color space, focusing on A and B channels differences to identify colored images, providing quicker analysis times.
