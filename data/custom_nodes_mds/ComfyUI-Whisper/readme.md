# ComfyUI Whisper

Transcribe audio and add subtitles to videos using [Whisper](https://github.com/openai/whisper/) in [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

![demo-image](https://github.com/yuvraj108c/ComfyUI-Whisper/blob/assets/recording.gif?raw=true)

## Installation

Install via [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)

## Usage

Load this [workflow](https://github.com/yuvraj108c/ComfyUI-Whisper/blob/master/example_workflows/whisper_video_subtitles_workflow.json) into ComfyUI & install missing custom nodes

## Nodes

### Apply Whisper

Transcribe audio and get timestamps for each segment and word.

### Add Subtitles To Frames

Add subtitles on the video frames. You can specify font family, font color and x/y positions.

### Add Subtitles To Background (Experimental)

Add subtitles like wordcloud on blank frames

## Credits

- [comfyanonymous/ComfyUI](https://github.com/comfyanonymous/ComfyUI)

- [Kosinkadink/ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)

- [melMass/comfy_mtb](https://github.com/melMass/comfy_mtb)
