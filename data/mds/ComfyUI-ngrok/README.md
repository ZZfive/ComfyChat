# ComfyUI-ngrok
Use ngrok to allow external access to ComfyUI.

*It is not a custom node.

## Installation: 

1. Use `git clone https://github.com/pkpkTech/ComfyUI-ngrok` in your ComfyUI custom nodes directory
2. Use `pip install -r requirements.txt` in ComfyUI-ngrok directory
3. Open the __init__.py file in a text editor, replace None on line 12 with your ngrok Authtoken and save.

If the preparation is successful, you will find the URL in the console when you start ComfyUI.

If this does not work, adding `--enable-cors-header` to the ComfyUI execution argument may help.
