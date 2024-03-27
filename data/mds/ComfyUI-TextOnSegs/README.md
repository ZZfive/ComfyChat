# ComfyUI-TextOnSegs
日本語版READMEは[こちら](README.jp.md)。  

<img src='img/example_face.jpg' width='400'>
<img src='img/example_board.jpg' width='400'>  
  
- Custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).
- Add a node for drawing text with [CR Draw Text](https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes/wiki/Text-Nodes#cr-draw-text) of [ComfyUI_Comfyroll_CustomNodes](https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes) to the area of SEGS detected by [Ultralytics Detector](https://github.com/ltdrdata/ComfyUI-extension-tutorials/blob/Main/ComfyUI-Impact-Pack/tutorial/detectors.md#ultralytics-detector) of [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack).

## Installation
```
cd <ComfyUI directory>/custom_nodes
git clone https://github.com/nkchocoai/ComfyUI-TextOnSegs.git
```

### Requirements
- [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)
- [ComfyUI_Comfyroll_CustomNodes](https://github.com/Suzie1/ComfyUI_Comfyroll_CustomNodes)

## Usage
- (Optional) Place font files (*.ttf) in the following folder.
  - `ComfyUI_windows_portable/ComfyUI/custom_nodes/ComfyUI_Comfyroll_CustomNodes/fonts`
  - Japanese and other characters are garbled, so you need to add font files.

### draw text on the face
- Load [workflows/draw_text_on_face.json](workflows/draw_text_on_face.json) with D&D.
- Change the values of Text node, CalcMaxFontSize node, etc. in the "Draw Text" group.
- Execute the workflow.
  - If detection fails, an error occurs, but this is a specification.

### Write text on the board
- Download [Can't show this \(meme\) SDXL](https://civitai.com/models/293531) and place it in the following folder.
  - `ComfyUI_windows_portable/ComfyUI/models/loras`
- Download [Board detector YOLO model \(For Can't show this \(meme\) SDXL\) \[Adetailer Model\] \- v1\.0](https://civitai.com/models/300228) and put it in the following folder.
  - `ComfyUI_windows_portable/ComfyUI/models/ultralytics/bbox`
- Load [workflows/draw_text_on_board.json](workflows/draw_text_on_board.json) with D&D.
- Change the values of the Text node, CalcMaxFontSize node, etc. in the "Draw Text" group.
- Execute the workflow.
  - If detection fails, an error occurs, but this is a specification.