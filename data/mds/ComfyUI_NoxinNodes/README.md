# ComfyUI_NoxinNodes
Utility nodes for ComfyUI that I created for me but am happy to share.

noxin_chimenode:
	This node grants the ability to trigger a sound file via the operating system. 
	It was designed to alert me that a long running ksampler batch or upscaler is done, and that there is something to review. I run from windows, so by default it points to the chimes.wav.
	I am aware of the security potential of running it via operating system subprocess, but the positive is that it could be repurposed to run a sorting/external upsampling system afterwards rather than play a sound.

noxin_scaledresolution:
	This node provides both raw and multiplied values of height and width, with a built in switch for SD1.5 and SDXL - I hated having to multiply them with multiple nodes and convert between floats and ints.

noxin_promptlibrary + noxin_saveprompt:
	Allows you to read from library text files in this folder from a single node, as well as save any non-existing prompts into that library via the save node. Try out the attached workflows for an example of use.

noxin_splitprompt:
	Organises the various pieces of your prompt into seperate sections and combines them according to the order string - I like it so I can quickly change just one element without hunting through test.

noxin_simplemath:
	does simple math and conversion things, takes in two ints, two floats and two strings, lets you add, subtract, multiply and divide (no error protection, watch those 0s) any pair, outputs to all three types. If you just want to do a conversion pass your value through with an add 0 operation.

Installation:
	To install this in your comfyui instance - navigate to the ComfyUI\custom_nodes folder, and either unzip or git clone this repo with command: git clone https://github.com/noxinias/ComfyUI_NoxinNodes.git
	Alternatively you can go to the code button at top, download as zip and place in the same location. The structure should look like ComfyUI\custom_nodes\ComfyUI_NoxinNodes\__init__.py (and other files)
	
Volitile Warning:
	I only just started making comfyui nodes, so there will be many updates as I figure out how to do things better. If you don't want to risk updates breaking your established workflow (basically replacing my nodes again) - You can delete the .git folder after install, to prevent auto-updates.