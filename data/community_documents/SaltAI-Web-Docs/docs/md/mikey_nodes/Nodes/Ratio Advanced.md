# Ratio Advanced (Mikey)
## Documentation
- Class name: `Ratio Advanced`
- Category: `Mikey/Utils`
- Output node: `False`

The Ratio Advanced node is designed to manage and apply various aspect ratios for image generation tasks. It enables the selection of predefined ratios, custom ratios, and adjustments to these ratios based on specific parameters, facilitating flexible and precise control over the dimensions of generated images.
## Input types
### Required
- **`preset`**
    - Allows the selection of a predefined ratio preset for image generation, offering a quick and standardized way to set aspect ratios.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`swap_axis`**
    - Determines whether the width and height of the selected ratio should be swapped, enabling the inversion of aspect ratios for different orientations.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`select_latent_ratio`**
    - Enables the selection of a specific ratio for the latent space dimensions, which influences the initial shape and size of the generated image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`custom_latent_w`**
    - Specifies a custom width for the latent space, allowing for precise control over the initial width of the generated image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`custom_latent_h`**
    - Specifies a custom height for the latent space, allowing for precise control over the initial height of the generated image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`select_cte_ratio`**
    - Allows the selection of a specific ratio for the constant tensor extension (CTE), which can affect the final dimensions of the generated image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`cte_w`**
    - Specifies a custom width for the constant tensor extension (CTE), enabling precise adjustments to the width of the generated image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cte_h`**
    - Specifies a custom height for the constant tensor extension (CTE), enabling precise adjustments to the height of the generated image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cte_mult`**
    - Applies a multiplier to the dimensions of the constant tensor extension (CTE), allowing for proportional scaling of the generated image's dimensions.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`cte_res`**
    - Sets a resolution for the constant tensor extension (CTE), which can influence the clarity and detail level of the generated image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`cte_fit_size`**
    - Adjusts the constant tensor extension (CTE) dimensions to fit within a specified size, ensuring the generated image meets certain size constraints.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`select_target_ratio`**
    - Enables the selection of a target ratio for the final image dimensions, guiding the aspect ratio of the output image.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`target_w`**
    - Specifies a custom width for the target image dimensions, allowing for precise control over the final width of the generated image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`target_h`**
    - Specifies a custom height for the target image dimensions, allowing for precise control over the final height of the generated image.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`target_mult`**
    - Applies a multiplier to the target image dimensions, enabling proportional scaling of the final image size.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`target_res`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`target_fit_size`**
    - unknown
    - Comfy dtype: `INT`
    - Python dtype: `unknown`
- **`crop_w`**
    - Specifies a custom width for cropping the generated image, allowing for precise control over the visible area of the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`crop_h`**
    - Specifies a custom height for cropping the generated image, allowing for precise control over the visible area of the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`use_preset_seed`**
    - Determines whether a preset seed should be used for image generation, ensuring reproducibility of results.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `List[str]`
- **`seed`**
    - Specifies a custom seed for the image generation process, allowing for controlled variation in the output.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`latent_w`**
    - Comfy dtype: `INT`
    - The width of the latent space after applying the selected ratios and adjustments.
    - Python dtype: `int`
- **`latent_h`**
    - Comfy dtype: `INT`
    - The height of the latent space after applying the selected ratios and adjustments.
    - Python dtype: `int`
- **`cte_w`**
    - Comfy dtype: `INT`
    - The width of the constant tensor extension (CTE) after applying the selected ratios and adjustments.
    - Python dtype: `int`
- **`cte_h`**
    - Comfy dtype: `INT`
    - The height of the constant tensor extension (CTE) after applying the selected ratios and adjustments.
    - Python dtype: `int`
- **`target_w`**
    - Comfy dtype: `INT`
    - The width of the target image dimensions after applying the selected ratios and adjustments.
    - Python dtype: `int`
- **`target_h`**
    - Comfy dtype: `INT`
    - The height of the target image dimensions after applying the selected ratios and adjustments.
    - Python dtype: `int`
- **`crop_w`**
    - Comfy dtype: `INT`
    - The width of the cropped area of the generated image, based on the specified cropping parameters.
    - Python dtype: `int`
- **`crop_h`**
    - Comfy dtype: `INT`
    - The height of the cropped area of the generated image, based on the specified cropping parameters.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class RatioAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        s.ratio_sizes, s.ratio_dict = read_ratios()
        default_ratio = s.ratio_sizes[0]
        # prepend 'custom' to ratio_sizes
        s.ratio_sizes.insert(0, 'custom')
        s.ratio_presets, s.ratio_config = read_ratio_presets()
        if 'none' not in s.ratio_presets:
            s.ratio_presets.append('none')
        return {"required": { "preset": (s.ratio_presets, {"default": "none"}),
                              "swap_axis": (['true','false'], {"default": 'false'}),
                              "select_latent_ratio": (s.ratio_sizes, {'default': default_ratio}),
                              "custom_latent_w": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                              "custom_latent_h": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                              "select_cte_ratio": (s.ratio_sizes, {'default': default_ratio}),
                              "cte_w": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                              "cte_h": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                              "cte_mult": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                              "cte_res": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                              "cte_fit_size": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                              "select_target_ratio": (s.ratio_sizes, {'default': default_ratio}),
                              "target_w": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                              "target_h": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                              "target_mult": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                              "target_res": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                              "target_fit_size": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                              "crop_w": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                              "crop_h": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                              "use_preset_seed": (['true','false'], {"default": 'false'}),
                              "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                              },
                "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO", "prompt": "PROMPT"}}

    RETURN_TYPES = ('INT', 'INT', # latent
                    'INT', 'INT', # clip text encode
                    'INT', 'INT', # target
                    'INT', 'INT') # crop
    RETURN_NAMES = ('latent_w', 'latent_h',
                    'cte_w', 'cte_h',
                    'target_w', 'target_h',
                    'crop_w', 'crop_h')
    CATEGORY = 'Mikey/Utils'
    FUNCTION = 'calculate'

    def mult(self, width, height, mult):
        return int(width * mult), int(height * mult)

    def fit(self, width, height, fit_size):
        if width > height:
            return fit_size, int(height * fit_size / width)
        else:
            return int(width * fit_size / height), fit_size

    def res(self, width, height, res):
        return find_latent_size(width, height, res)

    def calculate(self, preset, swap_axis, select_latent_ratio, custom_latent_w, custom_latent_h,
                  select_cte_ratio, cte_w, cte_h, cte_mult, cte_res, cte_fit_size,
                  select_target_ratio, target_w, target_h, target_mult, target_res, target_fit_size,
                  crop_w, crop_h, use_preset_seed, seed, unique_id=None, extra_pnginfo=None, prompt=None):
        # check if use_preset_seed is true
        if use_preset_seed == 'true' and len(self.ratio_presets) > 1:
            # seed is a randomly generated number that can be much larger than the number of presets
            # we use the seed to select a preset
            offset = seed % len(self.ratio_presets - 1)
            presets = [p for p in self.ratio_presets if p != 'none']
            preset = presets[offset]
        # check if ratio preset is selected
        if preset != 'none':
            latent_width = self.ratio_config[preset]['custom_latent_w']
            latent_height = self.ratio_config[preset]['custom_latent_h']
            cte_w = self.ratio_config[preset]['cte_w']
            cte_h = self.ratio_config[preset]['cte_h']
            target_w = self.ratio_config[preset]['target_w']
            target_h = self.ratio_config[preset]['target_h']
            crop_w = self.ratio_config[preset]['crop_w']
            crop_h = self.ratio_config[preset]['crop_h']
            if swap_axis == 'true':
                latent_width, latent_height = latent_height, latent_width
                cte_w, cte_h = cte_h, cte_w
                target_w, target_h = target_h, target_w
                crop_w, crop_h = crop_h, crop_w
            """
            example user_ratio_presets.json
            {
                "ratio_presets": {
                    "all_1024": {
                        "custom_latent_w": 1024,
                        "custom_latent_h": 1024,
                        "cte_w": 1024,
                        "cte_h": 1024,
                        "target_w": 1024,
                        "target_h": 1024,
                        "crop_w": 0,
                        "crop_h": 0
                    },
                }
            }
            """
            return (latent_width, latent_height,
                    cte_w, cte_h,
                    target_w, target_h,
                    crop_w, crop_h)
        # if no preset is selected, check if custom latent ratio is selected
        if select_latent_ratio != 'custom':
            latent_width = self.ratio_dict[select_latent_ratio]["width"]
            latent_height = self.ratio_dict[select_latent_ratio]["height"]
        else:
            latent_width = custom_latent_w
            latent_height = custom_latent_h
        # check if cte ratio is selected
        if select_cte_ratio != 'custom':
            cte_w = self.ratio_dict[select_cte_ratio]["width"]
            cte_h = self.ratio_dict[select_cte_ratio]["height"]
        else:
            cte_w = cte_w
            cte_h = cte_h
        # check if cte_mult not 0
        if cte_mult != 0.0:
            cte_w, cte_h = self.mult(cte_w, cte_h, cte_mult)
        # check if cte_res not 0
        if cte_res != 0:
            cte_w, cte_h = self.res(cte_w, cte_h, cte_res)
        # check if cte_fit_size not 0
        if cte_fit_size != 0:
            cte_w, cte_h = self.fit(cte_w, cte_h, cte_fit_size)
        # check if target ratio is selected
        if select_target_ratio != 'custom':
            target_w = self.ratio_dict[select_target_ratio]["width"]
            target_h = self.ratio_dict[select_target_ratio]["height"]
        else:
            target_w = target_w
            target_h = target_h
        # check if target_mult not 0
        if target_mult != 0.0:
            target_w, target_h = self.mult(target_w, target_h, target_mult)
        # check if target_res not 0
        if target_res != 0:
            target_w, target_h = self.res(target_w, target_h, target_res)
        # check if target_fit_size not 0
        if target_fit_size != 0:
            target_w, target_h = self.fit(target_w, target_h, target_fit_size)
        prompt.get(str(unique_id))['inputs']['output_latent_w'] = str(latent_width)
        prompt.get(str(unique_id))['inputs']['output_latent_h'] = str(latent_height)
        prompt.get(str(unique_id))['inputs']['output_cte_w'] = str(cte_w)
        prompt.get(str(unique_id))['inputs']['output_cte_h'] = str(cte_h)
        prompt.get(str(unique_id))['inputs']['output_target_w'] = str(target_w)
        prompt.get(str(unique_id))['inputs']['output_target_h'] = str(target_h)
        prompt.get(str(unique_id))['inputs']['output_crop_w'] = str(crop_w)
        prompt.get(str(unique_id))['inputs']['output_crop_h'] = str(crop_h)
        return (latent_width, latent_height,
                cte_w, cte_h,
                target_w, target_h,
                crop_w, crop_h)

```
