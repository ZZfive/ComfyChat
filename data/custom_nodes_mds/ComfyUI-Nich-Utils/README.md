# Nich Comfy Utils

**This library is still in early development! It will likely contain bugs or oddities**

This library contains utility (well, right now a single) nodes for comfyUI.

## Image from Dir Selector

This node will select a random image from the provided  folder. It allows for
- Pinning a single image (if you want to keep using the selected image).
- Allows searching within subdirectories of the given directory path (optional).
- Filtering on a regular expression, for if you want to sample specific files only (optional).

### Example Use cases:
- Selecting a random image for IPadapter
- Cycling controlnet poses
- Cycling only through a subset of images (regular expression knowledge required).

# Open work
- Improve code
- Add tests
- More nodes!
