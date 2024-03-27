# Comfy_custom_nodes
My custom nodes for ComfyUI. Just download the python script file and put inside ComfyUI/custom_nodes folder

<b>Prompt editing</b>

[a: b :step] --> replcae a by b at step

[a:step] --> add a at step

[a::step] --> remove a at step

<b>Alternating Tokens</b>

<a|b> --> alternate between a and b for all steps (could add more tokens)

note: to stop alternating at a certain step, use prompt editing e.g. an [<orange|apple>: apple :10] on a table

![Screenshot from 2023-08-06 09-25-42](https://github.com/taabata/Comfy_Syrian_Falcon_Nodes/assets/57796911/9f1ef805-22a9-4d37-89bd-24c10dea3374)





<b>Word as Image</b>

This node basically allows for user text input to be converted to an image of a black background and white text to be used with depth controlnet or T2I adapter models.

![Screenshot from 2023-06-17 20-45-55](https://github.com/taabata/Comfy_custom_nodes/assets/57796911/f782f336-053f-4d9a-b80c-dfbbf8bc34ab)
