# Style Conditioner Base Only (Mikey)
## Documentation
- Class name: `Style Conditioner Base Only`
- Category: `Mikey/Conditioning`
- Output node: `False`

The StyleConditionerBaseOnly node is designed to apply a specific style to a base conditioning without utilizing a seed for style selection. It focuses on modifying the base conditioning layers to reflect the chosen style, enhancing the overall aesthetic or thematic presentation of the content.
## Input types
### Required
- **`style`**
    - The 'style' parameter specifies the aesthetic or thematic style to be applied to the base conditioning. It plays a crucial role in determining the visual or thematic output of the node.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`strength`**
    - The 'strength' parameter controls the intensity of the style applied to the base conditioning, affecting how prominently the chosen style influences the final output.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`positive_cond_base`**
    - This parameter represents the positive aspect of the base conditioning that will be modified according to the selected style.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `torch.Tensor`
- **`negative_cond_base`**
    - This parameter represents the negative aspect of the base conditioning that will be modified according to the selected style.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `torch.Tensor`
- **`base_clip`**
    - The 'base_clip' parameter is used for encoding the style prompts before they are applied to the base conditioning.
    - Comfy dtype: `CLIP`
    - Python dtype: `CLIP`
- **`use_seed`**
    - This boolean parameter determines whether a seed should be used for selecting the style. If 'true', the style is selected based on the seed value; otherwise, the style is directly taken from the 'style' parameter.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `bool`
- **`seed`**
    - The 'seed' parameter is used in conjunction with 'use_seed' to deterministically select a style from a predefined list based on the seed value.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`base_pos_cond`**
    - Comfy dtype: `CONDITIONING`
    - Represents the modified positive aspect of the base conditioning after the application of the selected style.
    - Python dtype: `torch.Tensor`
- **`base_neg_cond`**
    - Comfy dtype: `CONDITIONING`
    - Represents the modified negative aspect of the base conditioning after the application of the selected style.
    - Python dtype: `torch.Tensor`
- **`style_str`**
    - Comfy dtype: `STRING`
    - A string indicating the specific style that has been applied to the base conditioning.
    - Python dtype: `str`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class StyleConditionerBaseOnly:
    @classmethod
    def INPUT_TYPES(s):
        s.styles, s.pos_style, s.neg_style = read_styles()
        return {"required": {"style": (s.styles,),"strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
                             "positive_cond_base": ("CONDITIONING",), "negative_cond_base": ("CONDITIONING",),
                             "base_clip": ("CLIP",),
                             "use_seed": (['true','false'], {'default': 'false'}),
                             "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                             }
        }

    RETURN_TYPES = ('CONDITIONING','CONDITIONING','STRING',)
    RETURN_NAMES = ('base_pos_cond','base_neg_cond','style_str',)
    FUNCTION = 'add_style'
    CATEGORY = 'Mikey/Conditioning'

    def add_style(self, style, strength, positive_cond_base, negative_cond_base,
                  base_clip,
                  use_seed, seed):
        if use_seed == 'true' and len(self.styles) > 0:
            offset = seed % len(self.styles)
            style = self.styles[offset]
        pos_prompt = self.pos_style[style]
        neg_prompt = self.neg_style[style]
        pos_prompt = pos_prompt.replace('{prompt}', '')
        neg_prompt = neg_prompt.replace('{prompt}', '')
        if style == 'none':
            return (positive_cond_base, negative_cond_base, style, )
        # encode the style prompt
        positive_cond_base_new = CLIPTextEncodeSDXL.encode(self, base_clip, 1024, 1024, 0, 0, 1024, 1024, pos_prompt, pos_prompt)[0]
        negative_cond_base_new = CLIPTextEncodeSDXL.encode(self, base_clip, 1024, 1024, 0, 0, 1024, 1024, neg_prompt, neg_prompt)[0]
        # average the style prompt with the existing conditioning
        positive_cond_base = ConditioningAverage.addWeighted(self, positive_cond_base_new, positive_cond_base, strength)[0]
        negative_cond_base = ConditioningAverage.addWeighted(self, negative_cond_base_new, negative_cond_base, strength)[0]
        return (positive_cond_base, negative_cond_base, style, )

```
