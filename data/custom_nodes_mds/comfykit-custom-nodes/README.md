# ComfyKit Custom Nodes

## LoraWithMetadata

Accepts 3 LoRAs to blend and is able to load trigger presets from text files
named the same as the LoRA file, but with `.tp01.txt`, `.tp02.txt`, etc, instead
of the `.safetensors`.

I like to copy the trigger words out of the website, put them in those presets
and then have a text widget show me all the triggers in play so I can work them
into my prompts.

TODO: make the preset number pull a line number instead. then just have
`.tp.txt` (trigger positive) and `.tn.txt` (trigger negative) files.

## TypecasterImage

Takes an image, returns the same image. Hard typecasted. This was to fix
problems in nodes that try too hard to be dynamic not knowing what the datatype
is when you first load a workflow.json, which required manual reconnection
to make the workflow work that first time.

TODO: make more typecasters. been abusing rg3 contexts for it but they also are
very bulky for many cases.
