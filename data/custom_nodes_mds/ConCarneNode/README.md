# ConCarneNode
ComfyUI Nodes

## Bing Image Grabber node

Collects a random image via Bing Image Search using input search term

Note - Collecting multiple images will crop and resize to 512x512 for batching
  
  
### Example usage -
  
  
**One image grabbed from bing to provide reference for img2img**

![Screenshot 2024-01-23 115918](https://github.com/concarne000/ConCarneNode/assets/49512595/f8657a76-d729-43a6-8d98-e428e2cae6eb)
![CC_00006_](https://github.com/concarne000/ConCarneNode/assets/49512595/ab13eed6-80ae-4573-95d2-24be4554533c)
(Output image contains workflow)<br><br><br>
  
  
**Using multiple images to embed IPAdapter on the fly**

Using a list of images of "woman drinking orange juice" from Bing, with the prompt of "selena gomez"

![Screenshot 2024-01-23 131310](https://github.com/concarne000/ConCarneNode/assets/49512595/a94137dc-8707-4a9a-8d5f-76c35629e7c7)
![CC_00027_](https://github.com/concarne000/ConCarneNode/assets/49512595/a67c25a3-d19d-4900-8758-17aace9e3a22)

(Output image contains workflow)<br><br><br>
  
  
**Using multiple images for facial embed**

Using a list of images of "PewDiePie" from Bing, cropped to face, with the prompt of "man in armor"


![Screenshot 2024-01-23 160004](https://github.com/concarne000/ConCarneNode/assets/49512595/19c6479a-ee08-4da0-baf0-22bd90060225)
![CC_00029_](https://github.com/concarne000/ConCarneNode/assets/49512595/692a28be-359f-474b-a066-3d072d70f65b)

(Output image contains workflow)<br><br><br>

## Zephyr node

Implements huggingface transformer of stability.ai's Zephyr 3B chat transformer to infer text based on prompt. Useful for creating a list of prompts.
