# ComfyUI-Anchors

A ComfyUI extension to add spatial anchors/waypoints to better navigate large workflows.

## Usage

Add Anchor Nodes (in `utils`)

Jump between them using the `a` and `d` keys.

### Unexpected side-effects you can abuse

The way Comfy's Litegraph's centerOnNode works doesn't account for collapsed nodes.\
So if you want to change the centering position when jumping to an Anchor, you can make it nice and big, then collapse it.\
Try it out.

## TODO

- [ ] Nodes
  - [x] Anchors
- [ ] Behavior
  - [x] `a` or `d` to jump between Anchors
  - [x] Anchors show their coordinates (Hacky widgets right now)
  - [ ] Show list of current Anchors (with names)
  - [ ] Allow changing of iteration order
    - [ ] Setting?
    - [ ] By name only?
  - [ ] Stretch: Add in vector based navigation, e.g. `w` would go to the anchor that most closely aligns with [0,1], `d` [0,-1], etc.
- [ ] Documentation

## Development

- Clone the repo
- `pnpm install`
- `pnpm build`

No plans to have this library work outside of ComfyUI

## Confessions

### Typing

To get the types working without extracting the JS code from ComfyUI, I have the development files (TS) reference Comfy as a sibling. So if you want TS hints and checks to work, you'll need to have a similar structure.

To get the JS to run properly, the relative path is swapped in the dist for a different relative path based on where the JS is copied on `__init__`.

Check out the `vite.config.ts` and `tsconfig.json` if you're interested. If you can find a cleaner/simpler way to do it, **please** tell me.
