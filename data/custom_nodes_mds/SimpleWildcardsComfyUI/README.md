# SimpleWildcardsComfyUI

A simple wildcard node for ComfyUI. Can also be used a style prompt node.

## SimpleConcat

Concats multiple text into one.

### inputs

- `delimeter(STRING)`: The delimeter to use
- `input_0-9`: The inputs to concat

### outputs

- `output_text(STRING)`: The generated string

## SimpleWildcard

### Where to place wildcard files

Place the wildcard files in the `wildcards` folder in the root of your comfyui directory.
The files should end in `.txt` and each value should be on it's own line.

### inputs

- `seed(INT)`: The seed to use to choose the wildcard.
- `control_after_generate`: What to do with the seed after queuing the prompt.
- `input_files(LIST)`: A glob for the input files to use.
- `input_text(LIST)`: Choose a line from the files yourself or select `*` to choose at random using seed.
- `prefix(STRING)`: The prefix to use
- `suffix(STRING)`: The suffix to use
- `regex(STRING)`: If provided this will be used to filter the input files.
- `temp_override(STRING)`: If not blank then overrides the output.
- `output_blank_ratio(FLOAT)`: A number between 0 and 1 to determine if the output should be blank. Not used if temp_override is set.

### outputs

- `output_text(STRING)`: The generated string
