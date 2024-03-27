# A ComfyUI Node for adding BLIP in CLIPTextEncode

## Announcement: [BLIP](https://github.com/salesforce/BLIP) is now officially integrated into CLIPTextEncode

### Dependencies
- [x] Fairscale>=0.4.4 (**NOT** in ComfyUI)
- [x] Transformers==4.26.1 (already in ComfyUI)
- [x] Timm>=0.4.12 (already in ComfyUI)
- [x] Gitpython (already in ComfyUI)

### Local Installation
Inside ComfyUI_windows_portable\python_embeded, run:
<pre>python.exe -m pip install fairscale</pre>

And, inside ComfyUI_windows_portable\ComfyUI\custom_nodes\, run:
<pre>git clone https://github.com/paulo-coronado/comfy_clip_blip_node</pre>

### Google Colab Installation
Add a cell with the following code:
<pre>
!pip install fairscale
!cd custom_nodes && git clone https://github.com/paulo-coronado/comfy_clip_blip_node
</pre>

### How to use
1. Add the CLIPTextEncodeBLIP node;
2. Connect the node with an image and select a value for min_length and max_length;
3. Optional: if you want to embed the BLIP text in a prompt, use the keyword **BLIP_TEXT** (e.g. "a photo of BLIP_TEXT", medium shot, intricate details, highly detailed).

### Acknowledgement
The implementation of **CLIPTextEncodeBLIP** relies on resources from <a href="https://github.com/salesforce/BLIP">BLIP</a>, <a href="https://github.com/salesforce/ALBEF">ALBEF</a>, <a href="https://github.com/huggingface/transformers">Huggingface Transformers</a>, and <a href="https://github.com/rwightman/pytorch-image-models/tree/master/timm">timm</a>. We thank the original authors for their open-sourcing.
