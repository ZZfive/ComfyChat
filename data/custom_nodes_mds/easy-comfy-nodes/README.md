# easy-comfy-nodes

A handful of utility nodes for ComfyUI

## Development

Clone this repo to your ComfyUI installation under `ComfyUI/custom_nodes/easy-comfy-nodes`, restart ComfyUI whenever you make a change
`__init__.py` is the entrypoint ComfyUI uses to discover the custom nodes, see [example_node in the comfy repo](https://github.com/comfyanonymous/ComfyUI/blob/eb5c991a8c24873b3efba747ec4466d4f2c986db/custom_nodes/example_node.py.example) for details

Prefix all easy-comfy-nodes with `EZ`

## Nodes

### Animation-related Nodes
- VideoCombine: Assembles an animated webp and uses the default provider chain to upload it to s3, returning the s3 url

### HTTP Nodes
Nodes for handling http requests as part of a workflow, these can be used to create webhooks and the like for different parts of a workflow

- HTTP POST: emits a POST request to `url` with dict `body` as JSON
- Load Img From URL: loads an image from a url
- Load Img Batch From URLs: loads a batch of images from a set of URLs on separate lines

### Dict Nodes
Some nodes for handling dictionary/map data structures

- Empty Dict: returns an empty Dict
- Assoc X: Associates `key` with `value` of type X