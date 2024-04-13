# Fe's Util nodes for ComfyUI

## Installation

### Manual installation

```bash
// switch to your project's root directory
cd custom_nodes
git clone https://github.com/fexli/fexli-util-node-comfyui.git
cd fexli-util-node-comfyui
pip install -r requirements.txt
```

### Installation via comfyui-manager

1. Open ComfyUI WebUI
2. Navigate to `Manager` -> `Install Custom Node`
3. Enter `fexli-util-node-comfyui` in the `Search` field, and click `Search`
4. Click `Install`

## Before you start

### configuration config.yaml

move `config_example.yaml` to `config.yaml` and fill in the following fields

```yaml
bc_docker_api: <bc docker infer api>
openai_key: <your openai key>
openai_host:
  - https://api.openai.com/v1/chat/completions
  - <your openai proxy>
```

Enjoy!
