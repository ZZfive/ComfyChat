# ModelMergeSDXL
## Documentation
- Class name: `ModelMergeSDXL`
- Category: `advanced/model_merging`
- Output node: `False`

The ModelMergeSDXL node specializes in merging two models by blending their components based on specified ratios. It allows for fine-tuned control over the merging process by adjusting the influence of each model's time embeddings, label embeddings, input blocks, middle block, output blocks, and output layer.
## Input types
### Required
- **`model1`**
    - The first model to be merged. It serves as the base model onto which patches from the second model are applied.
    - Comfy dtype: `MODEL`
    - Python dtype: `comfy.model_base.Model`
- **`model2`**
    - The second model to be merged. Key patches from this model are applied to the first model based on specified ratios.
    - Comfy dtype: `MODEL`
    - Python dtype: `comfy.model_base.Model`
- **`time_embed.`**
    - Specifies the ratio for blending the time embeddings of the two models.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`label_emb.`**
    - Specifies the ratio for blending the label embeddings of the two models.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`input_blocks.i`**
    - Specifies the ratio for blending the i-th input block of the two models. The index i ranges from 0 to 8.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`middle_block.i`**
    - Specifies the ratio for blending the i-th middle block of the two models. The index i ranges from 0 to 2.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`output_blocks.i`**
    - Specifies the ratio for blending the i-th output block of the two models. The index i ranges from 0 to 8.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`out.`**
    - Specifies the ratio for blending the output layer of the two models.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
## Output types
- **`model`**
    - Comfy dtype: `MODEL`
    - The merged model resulting from the blending of the two input models based on the specified ratios.
    - Python dtype: `comfy.model_base.Model`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ModelMergeSDXL(comfy_extras.nodes_model_merging.ModelMergeBlocks):
    @classmethod
    def INPUT_TYPES(s):
        arg_dict = { "model1": ("MODEL",),
                              "model2": ("MODEL",)}

        argument = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})

        arg_dict["time_embed."] = argument
        arg_dict["label_emb."] = argument

        for i in range(9):
            arg_dict["input_blocks.{}".format(i)] = argument

        for i in range(3):
            arg_dict["middle_block.{}".format(i)] = argument

        for i in range(9):
            arg_dict["output_blocks.{}".format(i)] = argument

        arg_dict["out."] = argument

        return {"required": arg_dict}

```
