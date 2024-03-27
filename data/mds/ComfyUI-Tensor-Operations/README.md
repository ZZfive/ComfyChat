# ComfyUI Tensor Operations Nodes

This repo contains nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that implement some helpful operations on tensors, such as normalization.

## Updates

### February 2nd, 2024

- Initial release. Added `Image Match Normalize` and `Latent Match Normalize` nodes.

## Nodes

### Image Match Normalize

This node returns a normalized version of the target image using the mean and standard deviation of each color channel of the source image. If you want the color and brightness of your image to match the colour and brightness of another image,
this is the node for you.

### Latent Match Normalize

This node returns a normalized version of the target latent using the mean and standard deviation of each channel of
the source latent. Latents encode color information differently than images and you may find that normalizing an image
by instead normalizing its latent representation results in a "better" result that is closer to the coloring of the
source.