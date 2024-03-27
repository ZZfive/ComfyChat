# Chatbox Overlay node for ComfyUI
## Custom node for ComfyUI to add a text box over a processed image before save node.

## Installation:

Download the py file and place it in the customnodes directory of your ComfyUI installation path.
Restart ComfyUI.

Add the node just before your save node by searching for "Chatbox Overlay".

Included is a sample chatbox for 1024x1024 images.
Adjust the start locations by calculating your image axis in pixels.
Adjust your font location. Ex. for windows it might be c:/windows/fonts/font.ttf

![image](https://github.com/Smuzzies/comfyui_chatbox_overlay/assets/110495122/62f54994-31f3-4846-9618-8b34336036e1)

## Workflow:
For the sample chatbox image included, save and load this image into ComfyUI.

![Filename Text 1_231012001707_Filename Text 2_00001_Filename Text 3](https://github.com/Smuzzies/comfyui_chatbox_overlay/assets/110495122/29aab687-e220-4c6b-8cf6-e999a0458c35)

![image](https://github.com/Smuzzies/comfyui_chatbox_overlay/assets/110495122/b633e129-94f2-4590-b54c-348d751dc0e0)

Example of stacking from X/Y coord calculations based of your image size.

![image](https://github.com/Smuzzies/comfyui_chatbox_overlay/assets/110495122/c1bfd52a-4b20-4111-9ede-d8ebc8fcc725)


## Credits to https://github.com/mikkel/ComfyUI-text-overlay and ChatGPT.
