# ComfyUI-PNG-Metadata Custom Nodes

<img src="examples/workflow.png"/>

ComfyUI-PNG-Metadata is a set of custom nodes for ComfyUI. It provides nodes that allow to add custom metadata to your PNG files, such as the prompt and settings used to generate the image. This also can be used to add "parameters" metadata item compatible with AUTOMATIC1111 metadata.

To use the nodes, simply add them to your workflow and connect the inputs. This will automatically add the metadata to any standard "save image" nodes.

The nodes provided in this library are:

1. **Set Metadata** - Set a single custom metadata field and optionally update 'parameters' field in png metadata.
2. **Set Metadata (All)** - Set multiple metadata at once and write them to 'parameters' field in png metadata.

*Note*: Currently the node execution order is not enforced, so if Metadata nodes execute after the Save Image node, the metadata will not be written to the image. This will be fixed in a future release.

## Installation

### Using ComfyUI-Manager
1. Install [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) if it isn't already.
2. Press Install Custom Nodes from the ComfyUI-Manager menu
3. Search for ComfyUI-PNG-Metadata
4. Click install

### Manual installation

Follow the steps below to install the ComfyUI-PNG-Metadata Library. These commands assume your current working directory is the ComfyUI root directory.

1. Clone the repository:
   ```
   git clone https://github.com/romeobuilderotti/ComfyUI-PNG-Metadata custom_nodes/ComfyUI-PNG-Metadata
   ```
2. Restart your ComfyUI.
3. You can find example workflows in the `custom_nodes/ComfyUI-PNG-Metadata/examples` directory.

