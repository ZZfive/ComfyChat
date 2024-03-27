# ComfyUI-Static-Primitives

Adds Static Primitives to ComfyUI. Mostly to work with reroute nodes

![Example](https://github.com/80sVectorz/ComfyUI-Static-Primitives/blob/main/images/Thumbnail.png?raw=true)

# Install instructions

Just clone this repo inside your custom_nodes folder: `git clone https://github.com/80sVectorz/ComfyUI-Static-Primitives.git`

# Usage

There should be a new node category called `primitives`.
There's a primitive for all the basic data-types.

Here's an example of the original [Sytan's SDXL 1.0 Workflow](https://github.com/SytanSD/Sytan-SDXL-ComfyUI) versus a version that's cleaned up using the static primitives:
![Example](https://github.com/80sVectorz/ComfyUI-Static-Primitives/blob/main/images/Original.png?raw=true)
![Example](https://github.com/80sVectorz/ComfyUI-Static-Primitives/blob/main/images/CleanedUpExample.png?raw=true)

## Collection primitives

Aside from the basic types included in this extension there's also a
collection type system.
This system allows users to add custom collection types.  
There are 2 included by default:
- samplers
- schedulers

More info about how you can add your own collection primitives can be found
[here](https://github.com/80sVectorz/ComfyUI-Static-Primitives/blob/main/collection_primitives/collection_primitives_readme.md).

# Rambling

I decided to install ComfyUI but got really discouraged by the wire spaghetti.
I thought, just having reroutes work with primitives would already
simplify avoiding spaghetti.
But, since reroutes don't work with primitives due to their dynamic nature.
I decided to create a set of static primitives.
These primitives have a predefined type and work with reroutes.
