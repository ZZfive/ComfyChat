# ⚡ ComfyUI Pronodes

- A collection of nice utility nodes for ComfyUI (still in development)
- Install via the [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager)

## Nodes List

### Load Youtube Video

- Downloads video from youtube url and saves to /output directory
- Automatically skips download if video already exists locally
- Displays the video and other details (fps/title) at the bottom of the node
- Seamless integration with [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)

### Preview VHS Audio

- Displays audio from [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)

### VHS Filenames to Path

- Convert Filenames to file path from [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)

## Changelog

- 08/03/2024 - Added preview VHS audio node + VHS Filenames to Path node
- 03/02/2024 - Refactor js with chaincallback (from VHS) + display video on reload
- 17/02/2024 - Added video + fps + title + resolution preview under LYV node
- 15/02/2024 - Simplify LYV node & remove VHS codes
- 14/02/2024 - Setup repository + publish to manager + Add Load Youtube Video Node (LYV)
