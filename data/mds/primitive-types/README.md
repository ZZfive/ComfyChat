# primitive-types
This repository contains typed primitives for ComfyUI in the form of the following nodes:
- `int`
- `float`
- `string`
- `string` (multiline)

The motivation for these primitives is that the standard primitive node cannot be routed. As a result, if you have a configuration node, e.g., `CFG`, you must connect it directly to the sampler node. However, with an `int` node, you can route it (several times if you wish)