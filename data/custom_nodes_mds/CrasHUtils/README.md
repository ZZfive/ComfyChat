# CrasH Utils Custom Nodes

## Image Glitcher

Based on the HTML image glitcher by Felix Turner [here](https://www.airtightinteractive.com/demos/js/imageglitcher/).

![image](https://github.com/chrish-slingshot/ComfyUI-ImageGlitcher/assets/117188274/b7b509a4-026e-4b03-98f3-70c10ec54a19)

### Usage

Should be fairly simple to use - simply plug in an image to the input.

- glitchiness - controls how much glitch corruption / chromatic abberation will be present
- brightness - brightens the image, useful when using the scanlines option
- scanlines - adds CRT TV style scanlines to the image

![ComfyUI_temp_cknmh_000362_](https://github.com/chrish-slingshot/ComfyUI-ImageGlitcher/assets/117188274/386ed082-d551-4520-ab9e-2c8fd4063f81)

## Color Stylizer

This node allows you to pick a single color in the image and greyscale everything else, keeping just the one color. It still needs a little work.

![srg_sdxl_preview_temp_utykd_00003_](https://github.com/chrish-slingshot/CrasHUtils/assets/117188274/828fe8f6-c225-490d-be60-820cfc73d1dd) ![ComfyUI_temp_tsubu_00004_](https://github.com/chrish-slingshot/CrasHUtils/assets/117188274/7faea8aa-b931-46f3-86b8-7b17432ad46e)

## Query Local LLM

Send a query to the [oobabooga](https://github.com/oobabooga/text-generation-webui) API which runs local LLMs, and then outputs the response. Make sure you've enabled the API option in your oobabooga session.

![image](https://github.com/chrish-slingshot/CrasHUtils/assets/117188274/c7070ce1-9823-48ba-ac13-135c5449b74a)

## SDXL Resolution Picker

Allows you to choose from the pre-trained resolutions that SDXL supports. There are other existing custom nodes that do this, but none that I could see had the full list of resolutions.

![image](https://github.com/chrish-slingshot/CrasHUtils/assets/117188274/4ac8d27c-4a6c-4ec0-b5c0-5e70ae7738ee) ![image](https://github.com/chrish-slingshot/CrasHUtils/assets/117188274/6e919da0-9f00-423d-b2da-6f7eabf11b62)

## SDXL Resolution Split

Converst the SDXL Resolution into width and height. This allows you to send your selected resolution around with just a single connection and then split it out where you need to, reducing the spaghetti lines.

![image](https://github.com/chrish-slingshot/CrasHUtils/assets/117188274/9b3cb55f-ecc8-444d-8657-74cf652c2fba)
