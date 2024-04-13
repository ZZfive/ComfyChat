# Colorize Facial Parts from PoseKPS
## Documentation
- Class name: `FacialPartColoringFromPoseKps`
- Category: `ControlNet Preprocessors/Pose Keypoint Postprocess`
- Output node: `False`

This node is designed to colorize specific facial parts in pose keypoints data based on provided colors for each facial part. It processes pose keypoints to highlight facial features such as eyes, nose, and lips by drawing them in specified colors, enhancing the visual representation of facial landmarks.
## Input types
### Required
- **`pose_kps`**
    - The pose keypoints data that contains information about the positions of various facial landmarks. This data is used to identify and colorize specific facial parts.
    - Comfy dtype: `POSE_KEYPOINT`
    - Python dtype: `List[Dict[str, Any]]`
- **`mode`**
    - Determines the drawing mode for facial parts. It can be either 'point' to draw points or 'polygon' to fill areas, affecting the visual representation of the colorized facial parts.
    - Comfy dtype: `COMBO[STRING]`
    - Python dtype: `str`
- **`skin`**
    - Specifies the color for the skin facial part. This affects the visual representation by coloring the skin area in the specified color.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`left_eye`**
    - Specifies the color for the left eye. This affects the visual representation by coloring the left eye area in the specified color.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`right_eye`**
    - Specifies the color for the right eye. This affects the visual representation by coloring the right eye area in the specified color.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`nose`**
    - Specifies the color for the nose. This affects the visual representation by coloring the nose area in the specified color.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`upper_lip`**
    - Specifies the color for the upper lip. This affects the visual representation by coloring the upper lip area in the specified color.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`inner_mouth`**
    - Specifies the color for the inner mouth. This affects the visual representation by coloring the inner mouth area in the specified color.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`lower_lip`**
    - Specifies the color for the lower lip. This affects the visual representation by coloring the lower lip area in the specified color.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - The output is a tensor of colorized pose frames, where specific facial parts have been highlighted in the provided colors.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class FacialPartColoringFromPoseKps:
    @classmethod
    def INPUT_TYPES(s):
        input_types = {
            "required": {"pose_kps": ("POSE_KEYPOINT",), "mode": (["point", "polygon"], {"default": "polygon"})}
        }
        for facial_part in FACIAL_PARTS: 
            input_types["required"][facial_part] = ("STRING", {"default": LAPA_COLORS[facial_part], "multiline": False})
        return input_types
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "colorize"
    CATEGORY = "ControlNet Preprocessors/Pose Keypoint Postprocess"
    def colorize(self, pose_kps, mode, **facial_part_colors):
        pose_frames = pose_kps
        np_frames = [self.draw_kps(pose_frame, mode, **facial_part_colors) for pose_frame in pose_frames]
        np_frames = np.stack(np_frames, axis=0)
        return (torch.from_numpy(np_frames).float() / 255.,)
            
    def draw_kps(self, pose_frame, mode, **facial_part_colors):
        canvas = np.zeros((pose_frame["canvas_height"], pose_frame["canvas_width"], 3), dtype=np.uint8)
        for (person, part_name) in itertools.product(pose_frame["people"], FACIAL_PARTS):
            facial_kps = rearrange(np.array(person['face_keypoints_2d']), "(n c) -> n c", n=70, c=3)[:, :2]
            facial_kps = facial_kps.astype(np.int32)
            part_color = ImageColor.getrgb(facial_part_colors[part_name])[:3]
            if mode == "circle":
                start, end = FACIAL_PART_RANGES[part_name]
                part_contours = facial_kps[start:end+1]
                for pt in part_contours:
                    cv2.circle(canvas, pt, radius=2, color=part_color, thickness=-1)
                continue

            if part_name not in ["upper_lip", "inner_mouth", "lower_lip"]:
                start, end = FACIAL_PART_RANGES[part_name]
                part_contours = facial_kps[start:end+1]
                if part_name == "skin":
                    part_contours[17:] = part_contours[17:][::-1]
            else:
                part_contours = facial_kps[FACIAL_PART_RANGES[part_name], :]
            cv2.fillPoly(canvas, pts=[part_contours], color=part_color)
        return canvas

```
