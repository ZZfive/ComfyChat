# ComfyUI-RAVE Attention
ComfyUI nodes to use RAVE attention as a temporal attention mechanism.

This differs from other implementations in that it does not concatenate the images together, but within the UNet's Self-Attention mechanism performs the RAVE technique.
By not altering the images/latents throughout the UNet, this method does not affect other temporal techniques, style mechanisms, or other UNet modifications.

For example, it can be combined with AnimateDiff, ModelScope/ZeroScope, or FLATTEN.

## Installation
This repository does not require any additional packages. Just git clone this into your `custom_nodes/` directory.

## Examples
See the `example_workflows` directory for example workflows.

## Acknowledgements
* [Ozgur Kara, Bariscan Kurtkaya, Hidir Yesiltepe, James M. Rehg, Pinar Yanardag](https://rave-video.github.io/) for researching and open sourcing their implementation of RAVE
