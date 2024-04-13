# MeshGraphormer Hand Refiner
## Documentation
- Class name: `MeshGraphormer-DepthMapPreprocessor`
- Category: `ControlNet Preprocessors/Normal and Depth Estimators`
- Output node: `False`

This node specializes in refining hand depth maps and predicting 2D joint positions in image space using the MeshGraphormer model. It processes input images to generate depth maps and masks, which are then used to refine hand representations and estimate joint positions, enhancing the accuracy of hand pose and shape analysis.
## Input types
### Required
- **`image`**
    - The input image or images to be processed for depth map and mask generation, serving as the primary data for hand analysis.
    - Comfy dtype: `IMAGE`
    - Python dtype: `torch.Tensor`
### Optional
- **`mask_bbox_padding`**
    - Determines the padding around detected hand bounding boxes, influencing the size and coverage of the output masks.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`resolution`**
    - The resolution to which the input images are resized before processing, affecting the detail level of the generated depth maps and masks.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`mask_type`**
    - Defines the method for generating masks, either based on depth information or tight bounding boxes, which impacts the segmentation quality of the hands in the input images.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`mask_expand`**
    - Specifies the expansion factor for masks, which influences the area covered by the masks and can affect the refinement process of the hand depth maps.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`rand_seed`**
    - A random seed to ensure reproducibility of the depth map and mask generation process, affecting the consistency of the output across multiple runs.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`IMAGE`**
    - Comfy dtype: `IMAGE`
    - The refined depth maps of hands, providing detailed depth information for further processing and analysis.
    - Python dtype: `torch.Tensor`
- **`INPAINTING_MASK`**
    - Comfy dtype: `MASK`
    - The generated masks for the input images, used to isolate and refine hand representations within the depth maps.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes:
    - [PreviewImage](../../Comfy/Nodes/PreviewImage.md)
    - [ControlNetApplyAdvanced](../../Comfy/Nodes/ControlNetApplyAdvanced.md)



## Source code
```python
class Mesh_Graphormer_Depth_Map_Preprocessor:
    @classmethod
    def INPUT_TYPES(s):
        types = create_node_input_types(mask_bbox_padding=("INT", {"default": 30, "min": 0, "max": 100}))
        types["optional"].update(
            {
                "mask_type": (["based_on_depth", "tight_bboxes", "original"], {"default": "based_on_depth"}),
                "mask_expand": ("INT", {"default": 5, "min": -MAX_RESOLUTION, "max": MAX_RESOLUTION, "step": 1}),
                "rand_seed": ("INT", {"default": 88, "min": 0, "max": 0xffffffffffffffff})
            }
        )
        return types

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("IMAGE", "INPAINTING_MASK")
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

    def execute(self, image, mask_bbox_padding=30, mask_type="based_on_depth", mask_expand=5, resolution=512, rand_seed=88, **kwargs):
        install_deps()
        from controlnet_aux.mesh_graphormer import MeshGraphormerDetector
        model = MeshGraphormerDetector.from_pretrained().to(model_management.get_torch_device())
        
        depth_map_list = []
        mask_list = []
        for single_image in image:
            np_image = np.asarray(single_image.cpu() * 255., dtype=np.uint8)
            depth_map, mask, info = model(np_image, output_type="np", detect_resolution=resolution, mask_bbox_padding=mask_bbox_padding, seed=rand_seed)
            if mask_type == "based_on_depth":
                H, W = mask.shape[:2]
                mask = cv2.resize(depth_map.copy(), (W, H))
                mask[mask > 0] = 255

            elif mask_type == "tight_bboxes":
                mask = np.zeros_like(mask)
                hand_bboxes = info["abs_boxes"]
                for hand_bbox in hand_bboxes: 
                    x_min, x_max, y_min, y_max = hand_bbox
                    mask[y_min:y_max+1, x_min:x_max+1, :] = 255 #HWC

            mask = mask[:, :, :1]
            depth_map_list.append(torch.from_numpy(depth_map.astype(np.float32) / 255.0))
            mask_list.append(torch.from_numpy(mask.astype(np.float32) / 255.0))
        depth_maps, masks = torch.stack(depth_map_list, dim=0), rearrange(torch.stack(mask_list, dim=0), "n h w 1 -> n 1 h w")
        return depth_maps, expand_mask(masks, mask_expand, tapered_corners=True)

```
