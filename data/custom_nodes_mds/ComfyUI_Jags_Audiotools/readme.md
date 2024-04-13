# 🎙️ComfyUI_Jags_Audiotools🎙️
A collection amazing audio tools for working with audio and sound files in comfyUI

✨🍬Please note the work in this repo is not completed and still in progress and code will break until all things are sorted. Please wait as will announce same in the update here. Thanks for your support. Also note for any issues on the nodes, kindly share your workflow and alos always update comfyUI base including all dependencies.✨🍬<br>

 "As on date some three nodes are having issues which is being resolved , especially load audio and play audio nodes"
 we are also adding functionality for LSP plugins and Open-Source Audio Plugins & Apps; if you have some cool suggestions would welcome creating a issue on same so can work on it to add same in. 

🌈 🍬 Note we have made changes to the original repo by **Diontimmer** here and thankful to him for making this possible , adding new nodes to experiment and certain features, experimental audio with contant updates. If you face any issue, ensure you have updated ComfyUi to the latest version and updates all requirements and share your workflow in the Issues. we are thankful to Harmonai-org and origianl repo for same🌈

# *Sample generator*  Tools to train a generative model on arbitrary audio samples

A big shout out and thanks to the wonderful work done on Sample diffusion by Harmonai and a lot of core code is ported from same to this repo and any queries on the working for same can be linked to the repo ass shown here. <a href ="https://github.com/Harmonai-org/sample-generator"> **sample-generator** </a>

### Features 🎺 🎸 🪕 🎻 🪘 🥁 🪗 🎤

Allows the use of trained dance diffusion/sample generator models in ComfyUI.

Also included are two optional extensions of the extension (lol); Wave Generator for creating primitive waves aswell as a wrapper for the Pedalboard library.

The pedalboard wrapper allows us to wrap most vst3s and control them, for now only a wrapper for OTT is included. Any suggestions are welcome.

Includes a couple of  helper functions.

### Nodes
<!-------------------------------------------------------------------------------------------------------------------------------------------------------->
<details>
    <summary><b> Jags_Audiotools </b> & <b>Node summary</b></summary>
      <ul>
        <li>Node that the gives user the ability to generate and play audio results through variety of different methods.</li>
        <li> Local models---The node pulls the required files from huggingface hub by default. You can create a models folder and place the modules there if you have a flaky connection or prefer to use it completely offline, it will load them locally instead. The path should be: ComfyUI/models/audio_diffusion; Alternatively, just clone the entire HF repo to it</li>
    </ul>
    <p align="center">
     -------- 
    </p>
</details>
     
<!-------------------------------------------------------------------------------------------------------------------------------------------------------->
### Example Work flow

<img src = images/2023-12-03_01-23-48.png >


### 🎸ComfyUI_Jags_Audiotools- WIKI

Link to the workflow and explanations : <a href ="https://github.com/jags111/ComfyUI_Jags_Audiotools/wiki"> **AUDIO TOOLS WIKI** </a>

### Dependencies


check Notes for more information.

## **Install:**
To install, drop the "_**ComfyUI_Jags_Audiotools**_" folder into the "_**...\ComfyUI\ComfyUI\custom_nodes**_" directory and restart UI.<br>

But the best method is to install same from ComfyUI Manager (https://github.com/ltdrdata/ComfyUI-Manager) and search for this name in the Node list and install from there and restart the UI as it takes care of all the dependencies and installs and make it easy for you. 

# Models - downloads and updates : 

Place models in ComfyUI/models/audio_diffusion ('model_folder' entry in config.yaml is accepted).

(Optional) Install xfer OTT VST3 from the website link [xfer OTT](https://xferrecords.com/freeware) <br>

**Link to the model collections:**

There are also a whole bunch of community-trained models on the :<a href =" https://discord.com/channels/1001555636569509948/1025191039352438794"> *Harmonai discord*</a> <br>

The original models from zqevans can be found in the <a href = "https://github.com/Harmonai-org/sample-generator/blob/main/Dance_Diffusion.ipynb">**Dance Diffusion notebook**</a> <br>

Most of the models are available for download from Huggingface page and it is easy to download same.
<a href = "https://huggingface.co/harmonai"> **Huggingface-Harmonai**</a> <br>




## Todo

[ ] Add guidance to notebook

## Acknowledgements

 - [sample-diffusion](https://github.com/sudosilico/sample-diffusion)
 - [pythongosssss](https://github.com/pythongosssss) 
 - [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
 - [Harmonai](https://github.com/Harmonai-org/sample-generator)
 - [pedalboard](https://github.com/spotify/pedalboard)
 - [xfer OTT](https://xferrecords.com/freeware)

# Comfy Resources

**ComfyUI_Jags_VectorMagic Linked Repos**
- [BlenderNeko ComfyUI_ADV_CLIP_emb](https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb)  by@BlenderNeko
- [Chrisgoringe cg-noise](https://github.com/chrisgoringe/cg-noise)  by@Chrisgoringe
- [pythongosssss ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts)  by@pythongosssss
- [shiimizu ComfyUI_smZNodes](https://github.com/shiimizu/ComfyUI_smZNodes)  by@shiimizu
- [LEv145_images-grid-comfyUI-plugin](https://github.com/LEv145/images-grid-comfy-plugin))  by@LEv145
- [ltdrdata-ComfyUI-Inspire-Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack) by@ltdrdata
- [pythongosssss-ComfyUI-custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts) by@pythongosssss
- [RockOfFire-ComfyUI_Comfyroll_CustomNodes](https://github.com/RockOfFire/ComfyUI_Comfyroll_CustomNodes) by@RockOfFire 

**Guides**:
- [Official Examples (eng)](https://comfyanonymous.github.io/ComfyUI_examples/)- 
- [ComfyUI Community Manual (eng)](https://blenderneko.github.io/ComfyUI-docs/) by @BlenderNeko

- **Extensions and Custom Nodes**:  
- [Plugins for Comfy List (eng)](https://github.com/WASasquatch/comfyui-plugins) by @WASasquatch
- [ComfyUI tag on CivitAI (eng)](https://civitai.com/tag/comfyui)-   
- [Tomoaki's personal Wiki (jap)](https://comfyui.creamlab.net/guides/) by @tjhayasaka

  ## Support
If you create a cool image with our nodes, please show your result and message us on twitter at @jags111 or @NeuralismAI .

You can join the <a href="https://discord.gg/vNVqT82W" alt="Neuralism Discord"> NEURALISM AI DISCORD </a> or <a href="https://discord.gg/UmSd4qyh" alt =Jags AI Discord > JAGS AI DISCORD </a> 
Share your work created with this model. Exchange experiences and parameters. And see more interesting custom workflows.

Support us in Patreon for more future models and new versions of AI notebooks.
- tip me on <a href="https://www.patreon.com/jags111"> [patreon]</a>

 My buymeacoffee.com pages and links are here and if you feel you are happy with my work just buy me a coffee !

 <a href="https://www.buymeacoffee.com/jagsAI"> coffee for JAGS AI</a> 

Thank you for being awesome! <br>

<img src = "images/CR2_up_00_00001_.png" width = "50%"> 

<!-- end support-pitch -->
