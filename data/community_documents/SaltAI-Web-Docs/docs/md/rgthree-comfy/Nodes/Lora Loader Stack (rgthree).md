# Lora Loader Stack (rgthree)
## Documentation
- Class name: `Lora Loader Stack (rgthree)`
- Category: `rgthree`
- Output node: `False`

The Lora Loader Stack (rgthree) node is designed to sequentially load multiple LoRA (Low-Rank Adaptation) modifications into a given model and CLIP instance. It allows for the dynamic enhancement or alteration of model behaviors by applying up to four distinct LoRA adjustments, each with its own strength level, thereby enabling fine-tuned control over the model's output characteristics.
## Input types
### Required
- **`model`**
    - The model to which LoRA modifications will be applied. It serves as the base for subsequent enhancements or alterations.
    - Comfy dtype: `MODEL`
    - Python dtype: `torch.nn.Module`
- **`clip`**
    - The CLIP instance to be modified alongside the model. It is used for text-image matching and can be adjusted with LoRA modifications.
    - Comfy dtype: `CLIP`
    - Python dtype: `torch.nn.Module`
- **`lora_01`**
    - The first LoRA modification to apply. It specifies the particular LoRA file to use for the adjustment.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`strength_01`**
    - The strength of the first LoRA modification. It determines the intensity of the adjustment applied to the model and CLIP.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`lora_02`**
    - The second LoRA modification to apply. Allows for additional adjustments beyond the first LoRA.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`strength_02`**
    - The strength of the second LoRA modification. Adjusts the model and CLIP with a specified intensity.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`lora_03`**
    - The third LoRA modification to apply. Enables further fine-tuning of the model and CLIP.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`strength_03`**
    - The strength of the third LoRA modification. Controls the level of adjustment applied.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`lora_04`**
    - The fourth and final LoRA modification to apply. Provides the last layer of adjustment.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`strength_04`**
    - The strength of the fourth LoRA modification. Determines the final adjustment intensity.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The model after applying all specified LoRA modifications. It reflects the cumulative adjustments made.
    - Python dtype: `torch.nn.Module`
- **`clip`**
    - Comfy dtype: `CLIP`
    - The CLIP instance after all LoRA modifications have been applied. It shows the adjusted text-image matching capabilities.
    - Python dtype: `torch.nn.Module`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [Text to Conditioning](../../was-node-suite-comfyui/Nodes/Text to Conditioning.md)
    - UltimateSDUpscale
    - Reroute
    - [KSampler](../../Comfy/Nodes/KSampler.md)
    - [BNK_CLIPTextEncodeAdvanced](../../ComfyUI_ADV_CLIP_emb/Nodes/BNK_CLIPTextEncodeAdvanced.md)
    - [KSamplerAdvanced](../../Comfy/Nodes/KSamplerAdvanced.md)
    - [CLIPTextEncode](../../Comfy/Nodes/CLIPTextEncode.md)
    - [CLIPSetLastLayer](../../Comfy/Nodes/CLIPSetLastLayer.md)
    - [ADE_AnimateDiffLoaderWithContext](../../ComfyUI-AnimateDiff-Evolved/Nodes/ADE_AnimateDiffLoaderWithContext.md)
    - [Anything Everywhere](../../cg-use-everywhere/Nodes/Anything Everywhere.md)



## Source code
```python
class RgthreeLoraLoaderStack:

    NAME = get_name('Lora Loader Stack')
    CATEGORY = get_category()

    @classmethod
    def INPUT_TYPES(cls):  # pylint: disable = invalid-name, missing-function-docstring
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP", ),

                "lora_01": (['None'] + folder_paths.get_filename_list("loras"), ),
                "strength_01":("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "lora_02": (['None'] + folder_paths.get_filename_list("loras"), ),
                "strength_02":("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "lora_03": (['None'] + folder_paths.get_filename_list("loras"), ),
                "strength_03":("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "lora_04": (['None'] + folder_paths.get_filename_list("loras"), ),
                "strength_04":("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"

    def load_lora(self, model, clip, lora_01, strength_01, lora_02, strength_02, lora_03, strength_03, lora_04, strength_04):
        if lora_01 != "None" and strength_01 != 0:
            model, clip = LoraLoader().load_lora(model, clip, lora_01, strength_01, strength_01)
        if lora_02 != "None" and strength_02 != 0:
            model, clip = LoraLoader().load_lora(model, clip, lora_02, strength_02, strength_02)
        if lora_03 != "None" and strength_03 != 0:
            model, clip = LoraLoader().load_lora(model, clip, lora_03, strength_03, strength_03)
        if lora_04 != "None" and strength_04 != 0:
            model, clip = LoraLoader().load_lora(model, clip, lora_04, strength_04, strength_04)

        return (model, clip)

```
