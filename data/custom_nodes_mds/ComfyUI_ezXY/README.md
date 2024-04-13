# ezXY

### A collection of scripts and custom nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

## Features
- Automatic type-casting for most custom and native nodes.
- Compact math node
- List generation for iteration control
- Versitile XY plotter 

## Installation
Clone the repository to `custom_nodes` in your ComfyUI directory:
```
git clone https://github.com/GMapeSplat/ComfyUI_ezXY.git
```

## Scripts
**Automatic type-casting**
>Patches Comfy UI during runtime to allow integer and float slots to connect. Data types are cast automatically and clamped to the input slot's configured minimum and maximum values. Should work out of the box with most custom and native nodes. Might cause some compatibility issues, or break depending on your version of ComfyUI.
>
>Runtime patches do not modify any source files, but it is fairly brute force. Use at your own risk. All patches can be deactivated by modifying the values in config.yaml.
## Nodes

*ezXY Driver*
- Simple list generator for quickly and easily setting up XY plot workflows. If used with other list generators or math nodes you can drive the primitive inputs of any node.

*ezXY Assemble Plot*
- Simple image plotter, now with labels. Works best with **ezXY Driver** and **String to Label** nodes.
  
*Plot Images*
- Formats image lists into a single image grid. Use with **ezXY Driver** to set up XY plots. Capable of plotting images of different sizes together.

*String to Label*
- Converts simple text into a white-on-black label.

*Join images*
- Joins two images together. If the images are not the same size, the smaller is padded to fit. Works well for applying labels to images.

*Item from Dropdown*
- When plugged into a converted dropdown input (COMBO) it populates with the list of options. Click to pick and order which options are prepared for execultion. Use with an iteration driver to compare models/loras/samplers.
  
*ezMath*
- Compact math node.
  
*Iteration Driver*
- Simple integer list generator. Outputs a list of integers within the specified range.
  
*Strings to List*
- Takes a string of semi-colon seperated items and outputs them as a list of strings.
  
*Numbers to List*
- Takes a string of semi-colon seperated numbers and outputs them as a list. Supports in-line math and
  ranges.

*String from List*
- Takes a string list as an input and outputs a single specified item.

*Number from string*
- Takes a list of numbers as an input and outputs a single specified item.
  
*Line to Console*
- Simple debugging node. Prints any input to the console.

## ToDo
- [x] Labeler node
- [x] List widget auto-populator? (for model loaders)
- [x] Workflow examples
- [ ] Add mask output to Image plotter
- [ ] Add comparison operators to math node
- [x] Add logic to type-casting patches to allow int/float to string conversions?
  