# Conditioning (Concat)
## Documentation
- Class name: `ConditioningConcat`
- Category: `conditioning`
- Output node: `False`

The ConditioningConcat node is designed to concatenate conditioning vectors, specifically merging the 'conditioning_from' vector into the 'conditioning_to' vector. This operation is fundamental in scenarios where the conditioning information from two sources needs to be combined into a single, unified representation.
## Input types
### Required
- **`conditioning_to`**
    - Represents the primary set of conditioning vectors to which the 'conditioning_from' vectors will be concatenated. It serves as the base for the concatenation process.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[torch.Tensor, Dict]]`
- **`conditioning_from`**
    - Consists of conditioning vectors that are to be concatenated to the 'conditioning_to' vectors. This parameter allows for additional conditioning information to be integrated into the existing set.
    - Comfy dtype: `CONDITIONING`
    - Python dtype: `List[Tuple[torch.Tensor, Dict]]`
## Output types
- **`conditioning`**
    - Comfy dtype: `CONDITIONING`
    - The output is a unified set of conditioning vectors, resulting from the concatenation of 'conditioning_from' vectors into the 'conditioning_to' vectors.
    - Python dtype: `Tuple[List[Tuple[torch.Tensor, Dict]]]`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [ControlNetApplyAdvanced](../../Comfy/Nodes/ControlNetApplyAdvanced.md)
    - [SamplerCustom](../../Comfy/Nodes/SamplerCustom.md)
    - [ConditioningConcat](../../Comfy/Nodes/ConditioningConcat.md)



## Source code
```python
class ConditioningConcat:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "conditioning_to": ("CONDITIONING",),
            "conditioning_from": ("CONDITIONING",),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "concat"

    CATEGORY = "conditioning"

    def concat(self, conditioning_to, conditioning_from):
        out = []

        if len(conditioning_from) > 1:
            print("Warning: ConditioningConcat conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

        cond_from = conditioning_from[0][0]

        for i in range(len(conditioning_to)):
            t1 = conditioning_to[i][0]
            tw = torch.cat((t1, cond_from),1)
            n = [tw, conditioning_to[i][1].copy()]
            out.append(n)

        return (out, )

```
