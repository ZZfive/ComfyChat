# SDFX config Documentation

## Introduction
This JSON configuration file contains settings and paths for the application.

**None of the parameters are mandatory**

## Global Parameters (`args`)
This section details each global parameter along with its meaning and possible values.
- `disable-xformers`: Boolean indicating whether xformers are disabled.
- `preview-method`: Method used for previewing.
- `listen`: IP address to listen for connections.
- `enable-cors-header`: Boolean indicating whether CORS headers are enabled.
- `port`: Port on which the service is listening.

For sdfx users using their own ComfyUI instance :

These parameters override the default flags passed to the main.py script upon launching ComfyUI.

```python main.py --disable-xformers --listen=127.0.0.1```

can be replaced with :

```python main.py --sdfx-config-file=/path/to/sdfx.config.json```


## Paths (`paths`)
This section describes the different paths used in the application.
- `media`: Paths to media directories.
  - `gallery`: Directory for media galleries.
  - `input`: Directory for media input files.
  - `output`: Output directory for processed media.
  - `workflows`: Directory for media workflows.
  - `templates`: Directory for media templates.
  - `temp`: Temporary directory for media.
- `models`: Paths to model directories.
  - `checkpoints`: Array of directories for model checkpoints.
  - `clip`: Array of directories for clip models.
  - `clip_vision`: Array of directories for clip vision models.
  - ...


All the paths can be specified as either relative or absolute.

If a path is defined as relative, it will be resolved relative to the location of the `sdfx.config.json` file.

For example, if the configuration includes:

```json
"checkpoints": [
    "data/models/checkpoints"
]
```
and the absolute path of sdfx.config.json is `/x/y/sdfx.config.json`, the checkpoints will be searched for at :
```/x/y/data/models/checkpoints```

## Usage Examples
If you don't want to change anything on your model path configuration or you are already using ComfyUI extra_model_paths.yaml to configure the path of your models you can simply delete everything from `sdfx.config.json` related to model path

So a minimalist configuration could be : 

```json
{ 
  "args": {
    "disable-xformers": true,
    "preview-method": "taesd",
    "listen": "127.0.0.1",
    "enable-cors-header": true,
    "port": 8188
  },

  "paths": {
    "media": {
      "gallery": "data/media/gallery",
      "input": "data/media/input",
      "output": "data/media/output",
      "workflows": "data/media/workflows",
      "templates":  "data/media/templates",
      "temp": "data/media/temp"
    }
  }
}
```