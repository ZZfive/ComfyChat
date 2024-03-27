# QRNG_Node_ComfyUI
A node that takes in an array of random numbers from the ANU QRNG API and stores them locally for generating quantum random number noise_seeds in ComfyUI

## Set Up:
The installation itself is pretty easy:
  1) Just grab the .py file from the repository and add it to your custom nodes folder in the ComfyUI file structure
  2) Visit https://quantumnumbers.anu.edu.au/ and get a free-tier API license
  3) Open qrng_node.py and add your API key to line 11
  4) Save the file

## Usage
Usage is also easy:
  1) Right click the KSampler or KSampler(Advanced)
  2) Select "Convert noise_seed to input"
  3) Add and connect the qrng node to the new noise_seed input
  4) Run generations like normal ^.^

## Note:
- The ANU QRNG API allows 100 requests per month
- I have each request grabbing 1024 16-bit integers
- I have the back-end doing some bit math to combine 4 of these into a 64-bit integer (the max size the noise_seed field can accept)
- This means that you get (100*1024)/4 or 25,600 quantum randomly generated seeds per month (not including increasing the batch size per queue because the subsequent generations are all based on the original randomly generated noise_seed, so ^ batch ^ possible generations per seed)
- I personally haven't come anywhere close to hitting this, but if people are using a means of looping generations so each loop uses it's own new seed, you might get closer to hitting the limit than I. At that point I'd reccommend upgrading your API key's tier.

## Aside
- I'm not in any way affiliated with ANU or their department providing this API. I just find it a fun tool to incorporate measurably random data into projects. AKA - I don't get paid for promoting their API :p

Link to ComfyUI: https://github.com/comfyanonymous/ComfyUI
Thanks Comfy and the community around this awesome UI! :3
