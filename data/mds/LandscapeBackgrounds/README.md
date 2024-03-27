This program is a custom node for comfyui which allows selective "randomness" in constructing a prompt for portrait backgrounds

put the python files into the comfyui custom nodes folder and restart comfyui to install

see screenshot for workflow example

when using the seed works a bit differently to usual. Basically there are 2 ways of using the node. Either put the seed mode to incremental. With this mode the node will generate a new prompt each time. The other mode is fixed. Using this mode you can generate multiple images for the same prompt. When you want to change it, you'll need to either change one of the fields or change the fixed seed number.

However remembering the seed value and expecting the node will produce the same prompt will not work as the node has its own random generator.

If the seed is set to 999 (Fixed) then the main portion of the prompt is set to 0 and the prompt will only include the pre_text and post_text.

that is all.
