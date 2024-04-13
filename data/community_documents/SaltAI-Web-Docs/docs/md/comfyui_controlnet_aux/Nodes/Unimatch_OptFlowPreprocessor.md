# Unimatch Optical Flow
## Documentation
- Class name: `Unimatch_OptFlowPreprocessor`
- Category: `ControlNet Preprocessors/Optical Flow`
- Output node: `False`

This node is designed for preprocessing optical flow data, specifically for enhancing and refining the flow estimations between frames in a video sequence. It employs advanced techniques such as flow propagation with self-attention and bilinear upsampling to achieve high-quality flow predictions, making it an essential component in video analysis and manipulation tasks.
## Input types
### Required
- **`ckpt_name`**
    - The checkpoint name for loading the pre-trained Unimatch model. This parameter is vital for initializing the model with weights that are optimized for optical flow estimation.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`backward_flow`**
    - A boolean flag indicating whether to compute the backward flow. This option allows for bidirectional flow estimation, enhancing the motion analysis by considering both forward and backward movements.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`bidirectional_flow`**
    - A boolean flag indicating whether to predict bidirectional flow. Enabling this option provides a more comprehensive understanding of motion by analyzing it in both directions.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`OPTICAL_FLOW`**
    - Comfy dtype: `OPTICAL_FLOW`
    - The estimated optical flow between consecutive frames, represented as a tensor. This output is crucial for understanding the motion dynamics within the video.
    - Python dtype: `torch.Tensor`
- **`PREVIEW_IMAGE`**
    - Comfy dtype: `IMAGE`
    - A visualization of the optical flow, useful for inspecting and debugging the flow predictions visually.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class Unimatch_OptFlowPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": dict(
                ckpt_name=(
                    ["gmflow-scale1-mixdata.pth", "gmflow-scale2-mixdata.pth", "gmflow-scale2-regrefine6-mixdata.pth"],
                    {"default": "gmflow-scale2-regrefine6-mixdata.pth"}
                ),
                backward_flow=("BOOLEAN", {"default": False}),
                bidirectional_flow=("BOOLEAN", {"default": False})
            )
        }

    RETURN_TYPES = ("OPTICAL_FLOW", "IMAGE")
    RETURN_NAMES = ("OPTICAL_FLOW", "PREVIEW_IMAGE")
    FUNCTION = "estimate"

    CATEGORY = "ControlNet Preprocessors/Optical Flow"

    def estimate(self, image, ckpt_name, backward_flow=False, bidirectional_flow=False):
        assert len(image) > 1, "[Unimatch] Requiring as least two frames as a optical flow estimator. Only use this node on video input."    
        from controlnet_aux.unimatch import UnimatchDetector
        tensor_images = image
        model = UnimatchDetector.from_pretrained(filename=ckpt_name).to(model_management.get_torch_device())
        flows, vis_flows = [], []
        for i in range(len(tensor_images) - 1):
            image0, image1 = np.asarray(image[i:i+2].cpu() * 255., dtype=np.uint8)
            flow, vis_flow = model(image0, image1, output_type="np", pred_bwd_flow=backward_flow, pred_bidir_flow=bidirectional_flow)
            flows.append(torch.from_numpy(flow).float())
            vis_flows.append(torch.from_numpy(vis_flow).float() / 255.)
        del model
        return (torch.stack(flows, dim=0), torch.stack(vis_flows, dim=0))

```
