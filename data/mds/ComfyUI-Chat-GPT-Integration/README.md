# Credits to Omar92. This is based heavily off of his code
https://github.com/omar92/ComfyUI-QualityOfLifeSuit_Omar92 

I rewrote the core logic because this code was not compaitible with the new version of the OpenAi api

# Updates
2//24/2024
-I have updated logic to get to the roles and the config json files to work with Linux and MacOs. 

2/21/2024
-I removed the text around the OpenAi response so now the response from chatgpt will go straight into the image generator. This makes roles useless so i will eventually remove that. 
  
1/4/2024 
-I removed max_words from the node as it didnt really seem to do anything. You can try inputting it in the prompt to see if that makes a difference. 

-Sometimes, for whatever reason, chat-gpt api just doesnt work so i added logic to try 3 times waiting 10 seconds in between each try. This seems to work 100% of the time

-Added a seed so that the prompt can be resubmitted to get a difference response. Make sure that the "control_after_generate" is set to random if you want this. if not you will use the same response the first time you submit a request to chat gpt

I plan to start working on a node to integrate with MistalAi. Studying for an interview right now so it's a bit difficult to find time. 


# ComfyUI-ChatGPTIntegration
Single node to prompt ChatGPT and will return an input for your CLip TEXT Encode Prompt

## ComfyUI
ComfyUI is an advanced node-based UI that utilizes Stable Diffusion, allowing you to create customized workflows such as image post-processing or conversions.

## How to install
Download the zip file.
Extract to ..\ComfyUI\custom_nodes.
Restart ComfyUI if it was running (reloading the web is not enough).
You will find my nodes under the new group O/....

## How to update
- No Auto Update

The file looks like this :

{
"openAI_API_Key": "sk-#################################"
}

## ChatGPTPrompt
This node harnesses the power of chatGPT, an advanced language model that can generate detailed image descriptions from a small input.
- you need to have  OpenAI API key , which you can find at https://beta.openai.com/docs/developer-apis/overview
- Once you have your API key, add it to the `config.json` file
- I have made it a separate file, so that the API key doesn't get embedded in the generated images.


## Contact
### Discord: vienteck#6218
### GitHub: vienteck (https://github.com/vienteck)
