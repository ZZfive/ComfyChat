# Vivax-Nodes

A collection of nodes I find useful.

## Nodes

### Chunk Up
Splits the input list or batch into chunks of the specified size.
This is usefull in combination with [For Loops](https://github.com/comfyanonymous/ComfyUI/pull/2666) to process large lists in smaller chunks. For example you can process 64 latents in batches of 32 if you dont have enough vram.

### Get chunk
Gets the chunk at the specified index from the input list.
The index input can be attached to the remaining output of a for loop (make sure "one_index" is enabled to use 1-based indexing in this case)

### Join Chunks
This takes in a chunk and a batch, and joins the chunk to the batch. this is useful at the end of the for loop to collect the results.

### Model from URL
Download the model from the specified URL to the output folder, and outputs it filename.
If no filename is provided the response headers, or url ending will be used.
The nodes checks if the file already exists, and if it does it will not download it again.

### Inspect
Prints the input to the console, for debugging purposes.

### Any String
Outputs the input string, but allowed to connect to anything, useful for example to specify the same lora model to multiple nodes. Or other string inputs that arent a generic string input.

