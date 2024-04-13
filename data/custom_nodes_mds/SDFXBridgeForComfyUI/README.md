# SDFXBridgeForComfyUI - ComfyUI Custom Node for SDFX Integration


## :exclamation: Important :exclamation:

If ComfyUI is not installed on your computer we strongly recommand you to follow the instructions directly on [SDFX](https://github.com/sdfxai/sdfx)
and close this page.

## Overview

SDFXBridgeForComfyUI is a custom node designed for seamless integration between ComfyUI and SDFX. This custom node allows users to make ComfyUI compatible with [SDFX](https://github.com/sdfxai/sdfx) when running the ComfyUI instance on their local machines.

## Dependency

Before proceeding with the installation, ensure that you have [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) installed as a custom node. This is a mandatory dependency for the proper functioning of SDFXBridgeForComfyUI.

To install the dependency, you can use the following command:

```bash
git clone https://github.com/ltdrdata/ComfyUI-Manager.git
cd ComfyUI-Manager && pip install -r requirements.txt
```

## Installation

1. Clone the repository into the ComfyUI custom_node directory:
    ```bash
    git clone https://github.com/sdfxai/SDFXBridgeForComfyUI.git
    ```

2. Install dependencies using pip:
    ```bash
    cd SDFXBridgeForComfyUI && pip install -r requirements.txt
    ```

3. Install [SDFX](https://github.com/sdfxai/sdfx) by following it's own documentation

## Configuration file

You will find a sample configuration file named `sdfx.config.example.json`.
This file is not mandatory.
If you want to us it, please rename it to `sdfx.config.json` and customize it. 

For detailed explainations on how this file works, please read the [doc](https://github.com/sdfxai/SDFXBridgeForComfyUI/blob/master/docs/sdfx_config.md)

## License

This project is licensed under the AGPL-3.0 license.

## Acknowledgments

Special thanks to the SDFX and ComfyUI communities for their support and collaboration.

