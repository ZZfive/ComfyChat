# ComfyUI-RefSampling

This is a proof-of-concept repo that implements:
* "Reference CNet" found in popular Image generation applications such as [Automatic11](https://github.com/Mikubill/sd-webui-controlnet/discussions/1236)
  * Allows image generation that adheres content to a reference image
* [Visual Style Prompting](https://curryjung.github.io/VisualStylePrompt/)
  * Allows image generation that adheres style to a reference image

Both of these techniques utilize attention injection from reference images into the sampled/generated image.

## Examples

### Text to Image Generation
Image Prompt: "orange gorilla stripes"

Visual Style Prompt: "black and orange stripes"

![ref_sampling](https://github.com/logtd/ComfyUI-RefSampling/assets/160989552/c88e5eb6-f33d-4e1f-aa58-65f9b82b4f7d)


### Image to Image Generation
Image Prompt: "an old man"

Visual Style Prompt "blue dragon, flames"

![input_image](https://github.com/logtd/ComfyUI-RefSampling/assets/160989552/abf8201e-ce05-401e-bee6-582e427aa68c)

![ref_sampling_img](https://github.com/logtd/ComfyUI-RefSampling/assets/160989552/cc2a0bca-a478-4406-b2be-732d36b9668b)
