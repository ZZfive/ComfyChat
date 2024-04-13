# Installation
Just put `hakkun_nodes.py` into `ComfyUI\custom_nodes`

Also avaliable to install by Manager - https://github.com/ltdrdata/ComfyUI-Manager
![image](https://github.com/tudal/Hakkun-ComfyUI-nodes/assets/799063/e13c6ef2-dd81-4e7a-8df9-6f87ef39fcdf)

Drag and drop ```hakkun_nodes_workflow.png``` into ComfyUI to check use example for all nodes.

Example is using ComfyUI-Custom-Scripts

https://github.com/pythongosssss/ComfyUI-Custom-Scripts

# Custom nodes

All nodes can be found in "Hakkun" category.

## Prompt parser
![image](https://github.com/tudal/Hakkun-ComfyUI-nodes/assets/799063/76e35cb4-fcf4-4d8e-915f-94ccd0d19471)

Allows you to write whole positive and negative prompt in one text field.

You can write on multiple lines. All lines will be joined by "," and cleaned of unnecessary white space.

### Special syntax examples:

Everything after ```@``` will be used as negative prompt.
```
3d render @ drawing, painting
```
positive: ```3d render```

negative: ```drawing, painting```

All lines starting with ```!``` will be ignored.

All lines starting with `?` will be randomly ignored with a default 50% chance.

You can modify the default 50% chance by specifying different percentages after `?`.

```
?20% explosion
```
This will give you a 20% chance to include the word 'explosion'.


To randomly select from a list, use this format:
```
[dog|cat|horse]
```

You can nest these lists on multiple levels:
```
[town|city] full of [[fast|slow] [cars|[bicycles|motorbikes]]|[[loud|cute] [horses|[ducks|chickens]]]]
```
This will generate sentences like:
```
town full of cute horses
city full of fast bicycles
```

To adjust the chances of selecting each element separately, assign weights using the following format:
```
[*150*car|*30*boat|train|*80*bike]
```
If a weight is not specified for an element, the default weight of 100 will be used. Weights can be placed anywhere within the elements.

This also works with nested elements:
```
[[*10*pink|blue] bedroom*30*|*120*city at [day|*150*night] with [cars|trains|*10*rockets]]
```
In this example, a pink bedroom will be very rare.

There's also the option to insert external text in ```<extra1>``` or ```<extra2>``` placeholders.

Include ```<extra1>``` and/or ```<extra2>``` anywhere in the prompt, and the provided text will be inserted before parsing. If you don't specify these triggers but provide text, it will be pasted at the beginning by default (as positive).

All occurrences of ```em:``` will be replaced with ```embedding:```.

If a line starts with ```^```, everything after that symbol and the **next lines** will be ignored.

All prompt between ```/*``` and ```*/``` will be ignored.

You can set the ```seed``` as an input to control randomness along with the rest of the workflow.

Using the ```debug``` output will provide you with all the information about the generated prompt.

### Tags:
![image](https://github.com/tudal/Hakkun-ComfyUI-nodes/assets/799063/18e66b51-57cc-408b-94ae-d658cfe663a1)

You can apply custom tags by ```tags``` string input or load text file from ```tags_file``` (full path)

Set tag by ```>>>YOUR_TAG``` + lines below as content (check workflow for example)

If you set tags by ```tags_file``` (full path) then it will be used instead of tags input.

You can use tags in tags only before you define those inner tags.

All tag lines will be joined into one line by ```,``` so:
 - don't use more that 1 negative ```@``` by tag
 - don't use ```!``` or ```?```

If you have pysssss nodes - setting tags file as default can be good idea (in comfyui settings):

![image](https://github.com/tudal/Hakkun-ComfyUI-nodes/assets/799063/1996c0e3-7972-4f63-8529-c121790e1558)

![image](https://github.com/tudal/Hakkun-ComfyUI-nodes/assets/799063/c073252f-3ce3-4c94-98fa-9ef6777302eb)


## Multi Text Merge
![image](https://github.com/tudal/Hakkun-ComfyUI-nodes/assets/799063/fbb83cf9-a715-45bd-b50e-ce1f9a6e9a21)

Allows to join multiple texts by specified delimiter. Put ```\n``` to use new line as delimiter.
No need to keep any order. Empty inputs will be ignored.

## Random Line
![image](https://github.com/tudal/Hakkun-ComfyUI-nodes/assets/799063/4f1575e9-06db-459a-b06d-b7608588d006)

Will output random line

## Random Line 4
![image](https://github.com/tudal/Hakkun-ComfyUI-nodes/assets/799063/8b6a5936-d56b-4fc0-8b0f-6b7453219f26)

Will output random lines from 4 textfields/inputs and join them by specified delimiter. Put ```\n``` to use new line as delimiter.

## Calculate Upscale
![image](https://github.com/tudal/Hakkun-ComfyUI-nodes/assets/799063/86b0e0b0-70b8-4f69-aba7-beb246f7a6b9)

Outputs target upscale float value based on input image height.
Also calculates tile size (with and height) for tools like UltimateSDUpscale by specified number of horizontal tiles.

## Image size to string
![image](https://github.com/tudal/Hakkun-ComfyUI-nodes/assets/799063/81dc5d21-f726-45f8-8d46-2ec17d16a6b7)

Outputs input image size in format: ```512x768```

## Any Converter
![image](https://github.com/tudal/Hakkun-ComfyUI-nodes/assets/799063/c3281a50-8873-4dd5-8f01-8ba347c0874c)

Universal primitive type converter. If string cant be number it wil be 0

## Image Resize To Height/Width
![image](https://github.com/tudal/Hakkun-ComfyUI-nodes/assets/799063/ec54d06d-41bd-451d-888d-5c52664edb80)

Resize image to desired width or height. Support batches. Uses lanczos.

## Load text
![image](https://github.com/tudal/Hakkun-ComfyUI-nodes/assets/799063/d1d6f04e-b1c5-44a6-83af-a2778d7e20ca)

Simplest load text file (somehow I could not find such stuff in other nodes that would work)

## Load Random Image
![image](https://github.com/tudal/Hakkun-ComfyUI-nodes/assets/799063/a3942c1d-2ed3-4823-9b3a-c13107ce58f7)

Will return one random image ('.jpg', '.jpeg', '.png', '.webp') from passed directory and its file name (without extension).

