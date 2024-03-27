# comfy_PoP

An ever-expanding list of custom nodes for https://github.com/comfyanonymous/ComfyUI/ that might be useful or possibly ridiculous. I've been making things I want to use, and I hope others find uses for them as well. 

To install please navigate to your ComfyUI custom_nodes folder. 
Then run git clone https://github.com/picturesonpictures/comfy_PoP

################################################################################################

My latest nodes modify conditioning pooled output. One multiplies and the other normalizes. If you've ever wanted to do either of these things, now you can. 

The multiplier will allow you to multiply by a negative number. I wouldn't suggest doing this and haven't found any use for doing such a thing, but you have the option. I do find it useful for adjusting the conditioning values with more complicated workflows, so if you stay in the positive values you might just find it useful.

The normalizer normalizes as the name implies. I haven't played with it enough to really say how useful normalization of the pooled output might be, but it does seem to tame things a bit.

I'd suggest just playing around with them and seeing what they can do. 

![image](https://github.com/picturesonpictures/comfy_PoP/assets/118248359/2735c5db-7c79-4faa-aff5-a71165941d05)

################################################################################################

I've added an Encoder and Decoder with built in VAE loaders so I didn't have to load them externally. As long as you use the same model in each one it'll only load that model into your memory once, so there's that. I realize there are other nodes like this, but I wanted to better understand how the encoding and decoding processes worked, so made my own.

![image](https://github.com/picturesonpictures/comfy_PoP/assets/118248359/d684897f-59e6-4d99-9b0b-f12ca0b3adae)

################################################################################################

AdaptiveCannyDetector detects edges and it's fast. It's somewhat experimental, didn't know how canny edge detection worked so I decided to make my own node and figure it out. I still don't really know how it works, but the node seems to work. I included some settings I hadn't seen in other comparable nodes. Fun to play with.

![image](https://github.com/picturesonpictures/comfy_PoP/assets/118248359/66141619-335f-475d-9ac7-459220b519db)


LoraStackLoader_PoP isn't exactly breaking new ground, but couldn't find a node that had all the attributes I was looking for so I put this together. 3 LoRAs, on-off switches, model and weight inputs and outputs. Simple and useful.

LoraStackLoader10_PoP, on the other hand, is a  bit ridiculous. Want to load 10 LoRAs in one place? Here's your chance. Each Slot comes with an on/off switch and it has identical inputs/outputs to the single LoRA loader.

![image](https://github.com/picturesonpictures/comfy_PoP/assets/118248359/b65e99a8-5aef-4e39-86cf-10879a986d20)


And finally, AnyAspectRatio is another simplish node. It converts the pixel count of a square image to pretty much any aspect ratio you want. You can also set the size of the square. This node will allow you to make very unreasonable things.

The default side length of 1024 (which corresponds to a 1-megapixel image) and rounding value of 64 are Stability recommended values for SDXL. if you're using SDXL I'd strongly recommend you not adjust these values. For 1.5 drop the side length to 512, and then if you're using one of the 2.1 models that are trained on 768x768 adjust accordingly. 

I question whether there's a viable use case for deviating from these values too much, but as AI enthusiast Mahatma Ghandi once said "Freedom is not worth having if it does not include the freedom to make mistakes."

![image](https://github.com/picturesonpictures/comfy_PoP/assets/118248359/54194301-fada-4700-a0a7-21d72223d641)

Do you like links? https://picturesonpictures.com/

the end
