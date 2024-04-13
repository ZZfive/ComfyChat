# RAFT Load Flow from EXR Channels
## Documentation
- Class name: `RAFTLoadFlowFromEXRChannels`
- Category: `jamesWalker55`
- Output node: `False`

This node is designed for loading motion flows from EXR image files, specifically tailored for use with Blender's vector pass in the Cycles renderer. It facilitates the extraction and combination of specific color channels from the vector pass to generate motion flow data.
## Input types
### Required
- **`path`**
    - Specifies the file path to the EXR image from which motion flows are to be loaded. It is crucial for locating and accessing the image file.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`x_channel`**
    - Determines the color channel (R, G, B, A) used for the x-component of the motion flow. This choice affects how motion is interpreted horizontally.
    - Comfy dtype: `['R', 'G', 'B', 'A']`
    - Python dtype: `str`
- **`y_channel`**
    - Determines the color channel (R, G, B, A) used for the y-component of the motion flow. This choice affects how motion is interpreted vertically.
    - Comfy dtype: `['R', 'G', 'B', 'A']`
    - Python dtype: `str`
- **`invert_x`**
    - Indicates whether to invert the motion flow in the x-direction. This can be necessary depending on the orientation of the motion in the source image.
    - Comfy dtype: `['false', 'true']`
    - Python dtype: `bool`
- **`invert_y`**
    - Indicates whether to invert the motion flow in the y-direction. This can be necessary depending on the orientation of the motion in the source image.
    - Comfy dtype: `['false', 'true']`
    - Python dtype: `bool`
## Output types
- **`raft_flow`**
    - Comfy dtype: `RAFT_FLOW`
    - The output is a motion flow data structure compatible with the RAFT algorithm. It represents the extracted motion flow from the EXR image.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
@register_node("RAFTEstimate", "RAFT Estimate")
class _:
    """
    https://pytorch.org/vision/main/auto_examples/plot_optical_flow.html
    """

    CATEGORY = "jamesWalker55"
    INPUT_TYPES = lambda: {
        "required": {
            "image_a": ("IMAGE",),
            "image_b": ("IMAGE",),
        }
    }
    RETURN_TYPES = ("RAFT_FLOW",)
    FUNCTION = "execute"

    def execute(self, image_a: torch.Tensor, image_b: torch.Tensor):
        """
        Code derived from:
        https://pytorch.org/vision/main/auto_examples/plot_optical_flow.html
        """

        assert isinstance(image_a, torch.Tensor)
        assert isinstance(image_b, torch.Tensor)

        torch_device = model_management.get_torch_device()
        offload_device = model_management.unet_offload_device()

        image_a = comfyui_to_native_torch(image_a).to(torch_device)
        image_b = comfyui_to_native_torch(image_b).to(torch_device)
        model = load_model().to(torch_device)

        image_a = preprocess_image(image_a)
        image_b = preprocess_image(image_b)

        all_flows = model(image_a, image_b)
        best_flow = all_flows[-1]
        # best_flow.shape => torch.Size([1, 2, 512, 512])

        model.to(offload_device)
        image_a = image_a.to("cpu")
        image_b = image_b.to("cpu")
        best_flow = best_flow.to("cpu")

        return (best_flow,)

```
