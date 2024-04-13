# unwdef Custom Nodes for ComfyUI

At the moment, only one node is available.

## Randomize LoRAs Node
The Randomize LoRAs node randomly loads LoRAs based on a predefined selection with also randomized weights. This enables users to experiment with different artistic effects on their generated images.

![preview](https://github.com/unwdef/unwdef-nodes-comfyui/assets/166751903/686f12e1-ed35-4165-94f7-048c0550c2fc)
Note: The "Show Text" node is part of [pythongosssss/ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts)

### How It Works
Connect the **model** and **clip** outputs from this node to your KSampler or other processing nodes. The output, **chosen loras**, provides a textual representation detailing which LoRAs and corresponding weights were applied during the generation.

### Configuration Fields
- **seed**: Ensures reproducibility. Maintain the same seed for consistent results across generations. _Note: Keep the same selected loras for this to work._
- **max_random**: Limits the maximum number of LoRAs to apply. Even if you select up to 10, you can choose to apply fewer.
- **lora_x**: Specifies the LoRA file to use.
- **min_str_x** and **max_str_x**: Defines the minimum and maximum strengths for each LoRA, allowing for a range of intensities.

### Installation
To install the Randomize LoRAs node in ComfyUI:

1. Open your terminal and navigate to your `ComfyUI/custom_nodes` directory.
2. Clone the repository using:
   ```
   git clone https://github.com/unwdef/unwdef-nodes-comfyui.git
   ```
3. Restart ComfyUI to apply the changes.  

### Uninstallation
To remove the custom node:
1. Delete the `unwdef-nodes-comfyui` directory from `ComfyUI/custom_nodes`.
2. Restart ComfyUI to apply the changes. 

### Updates
To update the node:

1. Navigate to `ComfyUI/custom_nodes/unwdef-nodes-comfyui` in your terminal.
2. Run the following command: `git pull`
3. Restart ComfyUI to apply the changes. 
