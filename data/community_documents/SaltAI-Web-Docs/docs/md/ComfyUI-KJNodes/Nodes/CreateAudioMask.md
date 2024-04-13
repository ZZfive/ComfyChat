# CreateAudioMask
## Documentation
- Class name: `CreateAudioMask`
- Category: `KJNodes/masking/generate`
- Output node: `False`

The CreateAudioMask node generates audio-based masks for images by converting audio signals into visual representations. It utilizes spectrogram data from audio files to create circular masks that vary in size based on the audio's amplitude, allowing for dynamic visual effects in image processing.
## Input types
### Required
- **`invert`**
    - A boolean flag that, when set to True, inverts the color of the generated masks, offering an alternative visual effect.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
- **`frames`**
    - Specifies the number of frames (images) to generate masks for, affecting the batch size and the output array dimensions.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`scale`**
    - A scaling factor applied to the radius of the circular masks, allowing for customization of the mask size relative to the audio amplitude.
    - Comfy dtype: `FLOAT`
    - Python dtype: `float`
- **`audio_path`**
    - The file path to the audio file used for generating the masks, crucial for extracting the audio's spectrogram data.
    - Comfy dtype: `STRING`
    - Python dtype: `str`
- **`width`**
    - Determines the width of the generated images, influencing the dimensions of the output masks.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`height`**
    - Sets the height of the generated images, impacting the size and shape of the output masks.
    - Comfy dtype: `INT`
    - Python dtype: `int`
## Output types
- **`image`**
    - Comfy dtype: `IMAGE`
    - Returns the generated image masks as a tensor. If the 'invert' input parameter is set to True, the colors of the masks are inverted, providing an alternative visual effect.
    - Python dtype: `torch.Tensor`
## Usage tips
- Infra type: `GPU`
- Common nodes: unknown


## Source code
```python
class CreateAudioMask:
    def __init__(self):
        try:
            import librosa
            self.librosa = librosa
        except ImportError:
            print("Can not import librosa. Install it with 'pip install librosa'")
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "createaudiomask"
    CATEGORY = "KJNodes/masking/generate"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "invert": ("BOOLEAN", {"default": False}),
                 "frames": ("INT", {"default": 16,"min": 1, "max": 255, "step": 1}),
                 "scale": ("FLOAT", {"default": 0.5,"min": 0.0, "max": 2.0, "step": 0.01}),
                 "audio_path": ("STRING", {"default": "audio.wav"}),
                 "width": ("INT", {"default": 256,"min": 16, "max": 4096, "step": 1}),
                 "height": ("INT", {"default": 256,"min": 16, "max": 4096, "step": 1}),
        },
    } 

    def createaudiomask(self, frames, width, height, invert, audio_path, scale):
             # Define the number of images in the batch
        batch_size = frames
        out = []
        masks = []
        if audio_path == "audio.wav": #I don't know why relative path won't work otherwise...
            audio_path = os.path.join(script_dir, audio_path)
        audio, sr = self.librosa.load(audio_path)
        spectrogram = np.abs(self.librosa.stft(audio))
        
        for i in range(batch_size):
           image = Image.new("RGB", (width, height), "black")
           draw = ImageDraw.Draw(image)
           frame = spectrogram[:, i]
           circle_radius = int(height * np.mean(frame))
           circle_radius *= scale
           circle_center = (width // 2, height // 2)  # Calculate the center of the image

           draw.ellipse([(circle_center[0] - circle_radius, circle_center[1] - circle_radius),
                      (circle_center[0] + circle_radius, circle_center[1] + circle_radius)],
                      fill='white')
             
           image = np.array(image).astype(np.float32) / 255.0
           image = torch.from_numpy(image)[None,]
           mask = image[:, :, :, 0] 
           masks.append(mask)
           out.append(image)

        if invert:
            return (1.0 - torch.cat(out, dim=0),)
        return (torch.cat(out, dim=0),torch.cat(masks, dim=0),)

```
