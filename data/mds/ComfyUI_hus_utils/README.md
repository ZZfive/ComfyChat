# hus' utils for ComfyUI
Some nodes I cobbled together to satisfy my preferences, to recreate some behaviours of A1111.

## Nodes

### Fetch widget value
Extracted from 'Math Expression' from [ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts).

Fetch the value of widget _widget_name_ from node _node_name_. Node name can be type, title or S&amp;R name. If _multiple_ is 'no', use first matching node, if it is 'yes', return all as concatenated string separated by ', '.

### 3way Prompt Styler
Adapted from [Load Styles CSV](https://github.com/theUpsider/ComfyUI-Styles_CSV_Loader). Positive split into G and L for SDXL.

Load prompt styles from a file named 'styles.csv' in the source directory of this node (usually .../ComfyUI/custom_nodes/ComfyUI_hus_utils/). Each line has four columns: _style name_, _prompt G_, _prompt L_ and _negative prompt_. L and negative are simply appended to the respective inputs, the value of G is searched for '{prompt}', which is replaced with the G input.

Example provided in _example styles.csv_

### Text Hash
Return the first _length_ characters of the sha256 digest of the input.

### Date Time Format
Return the current date and/or time as formatted by strftime using _format_.

### Batch State
Check whether the state of the workflow (inputs and linking) has changed.

- _changed_ returns 1 if a change is detected, 0 otherwise
- _hash_ returns a hash of the current state
- _count_ counts up every time the state is unchanged and resets to 0 on change

Add _count_ to a seed input with 'fixed' value to create a sequence of images with consecutive seeds every time 'Queue Prompt' is clicked. When any input is changed, the sequence starts again from the value of the seed node. The purpose is to check the influence of changes in prompt, CFG, steps etc on the same set of images.

If the seed node is set to 'randomize' or anything else except 'fixed', it's input value will change every time, causing _count_ to remain at 0, which should cause the sum to show the expected behaviour.

### Debug Extra
Expose some of the internal workings for developing custom nodes.

## Example
Example workflow based on [Sytan SDXL ComfyUI](https://github.com/SytanSD/Sytan-SDXL-ComfyUI)

- Filename prefixes are created as _prompt hash_-_prompt style_-_seed_-_model name_-_base {sampler, scheduler, steps, cfg}_-_refiner {sampler, scheduler, steps, cfg}_
- Seeds are generated in a fixed sequence which restarts every time any input is changed.

### Dependencies
Nodes from ...

- [WAS Node Suite](https://github.com/WASasquatch/was-node-suite-comfyui)
- [ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts)
- [tinyterraNodes](https://github.com/TinyTerra/ComfyUI_tinyterraNodes)
- and probably some more I forgot ...
