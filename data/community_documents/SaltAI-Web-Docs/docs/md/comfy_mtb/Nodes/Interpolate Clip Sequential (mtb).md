# Interpolate Clip Sequential (mtb)
## Documentation
- Class name: `Interpolate Clip Sequential (mtb)`
- Category: `mtb/conditioning`
- Output node: `False`

This node is designed for the sequential interpolation of text encodings within a given base text, allowing for dynamic text manipulation by replacing specified segments with interpolated encodings. It leverages clip information and interpolation strength to achieve nuanced text transformations.
## Input types
### Required
- **`base_text`**
    - The original text from which specific segments will be replaced with interpolated encodings. It serves as the foundational content for the interpolation process.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`text_to_replace`**
    - The segment of the base text that is targeted for replacement through interpolation. This parameter identifies the specific portion of text to be dynamically altered.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`clip`**
    - A clip encoding that guides the interpolation process, ensuring that the text replacement aligns with the desired semantic direction or theme.
    - Comfy dtype: `CLIP`
    - Python dtype: `torch.Tensor`
- **`interpolation_strength`**
    - Determines the degree to which the replacement text is blended with the original, allowing for fine-tuned control over the interpolation effect.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The output is a conditioning format that includes interpolated text encodings, suitable for further processing or application in text-to-image or other generative models.
    - Python dtype: `List[Tuple[torch.Tensor, Dict[str, Any]]]`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class InterpolateClipSequential:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_text": ("STRING", {"multiline": True}),
                "text_to_replace": ("STRING", {"default": ""}),
                "clip": ("CLIP",),
                "interpolation_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "interpolate_encodings_sequential"

    CATEGORY = "mtb/conditioning"

    def interpolate_encodings_sequential(
        self, base_text, text_to_replace, clip, interpolation_strength, **replacements
    ):
        log.debug(f"Received interpolation_strength: {interpolation_strength}")

        # - Ensure interpolation strength is within [0, 1]
        interpolation_strength = max(0.0, min(1.0, interpolation_strength))

        # - Check if replacements were provided
        if not replacements:
            raise ValueError("At least one replacement should be provided.")

        num_replacements = len(replacements)
        log.debug(f"Number of replacements: {num_replacements}")

        segment_length = 1.0 / num_replacements
        log.debug(f"Calculated segment_length: {segment_length}")

        # - Find the segment that the interpolation_strength falls into
        segment_index = min(
            int(interpolation_strength // segment_length), num_replacements - 1
        )
        log.debug(f"Segment index: {segment_index}")

        # - Calculate the local strength within the segment
        local_strength = (
            interpolation_strength - (segment_index * segment_length)
        ) / segment_length
        log.debug(f"Local strength: {local_strength}")

        # - If it's the first segment, interpolate between base_text and the first replacement
        if segment_index == 0:
            replacement_text = list(replacements.values())[0]
            log.debug("Using the base text a the base blend")
            # -  Start with the base_text condition
            tokens = clip.tokenize(base_text)
            cond_from, pooled_from = clip.encode_from_tokens(tokens, return_pooled=True)
        else:
            base_replace = list(replacements.values())[segment_index - 1]
            log.debug(f"Using {base_replace} a the base blend")

            # - Start with the base_text condition replaced by the closest replacement
            tokens = clip.tokenize(base_text.replace(text_to_replace, base_replace))
            cond_from, pooled_from = clip.encode_from_tokens(tokens, return_pooled=True)

            replacement_text = list(replacements.values())[segment_index]

        interpolated_text = base_text.replace(text_to_replace, replacement_text)
        tokens = clip.tokenize(interpolated_text)
        cond_to, pooled_to = clip.encode_from_tokens(tokens, return_pooled=True)

        # - Linearly interpolate between the two conditions
        interpolated_condition = (
            1.0 - local_strength
        ) * cond_from + local_strength * cond_to
        interpolated_pooled = (
            1.0 - local_strength
        ) * pooled_from + local_strength * pooled_to

        return ([[interpolated_condition, {"pooled_output": interpolated_pooled}]],)

```
