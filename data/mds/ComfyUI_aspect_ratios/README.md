# ComfyUI_aspect_ratios

| [English](README.md) | [日本語](README-jp.md) |

I created an aspect ratio selector for ComfyUI based on [sd-webui-ar](https://github.com/alemelis/sd-webui-ar?tab=readme-ov-file).

![image](img1.png)

# Usage

Replace `Empty Latent Image` with `Aspect Ratios Node`.

- `size`: Reference size
- `aspect_ratios`: Set aspect ratios
- `standard`: Choose whether the reference size is based on width or height
- `swap_aspect_ratio`: Swap aspect ratios (change 1:2 to 2:1, for example)
- `batch_size`: Number of images to create

## Configuration

A configuration file, `aspect_ratios.txt`, will be created in `ComfyUI\custom_nodes\ComfyUI_aspect_ratios`.

Configure the aspect ratios in the file `aspect_ratios.txt`, following the specified format. For example:
```aspect_ratios.txt
1:1, 1/1 # 1:1 ratio based on minimum dimension
3:2, 3/2 # Set width based on 3:2 ratio to height
4:3, 4/3 # Set width based on 4:3 ratio to height
16:9, 16/9 # Set width based on 16:9 ratio to height
1.618:1, 1.618/1
# 1.414:1, 1.414/1
```
- The initial `1:1` is what is displayed in the UI.
- The next `1/1` is the value processed internally.
    - Do not include any characters other than `numbers` and `/`.
- The last line, `# 1:1 ratio based on minimum dimension`, is a comment.

The line starting with `#` is a comment and will not be read.

## Example

![image](img2.png)

Setting `standard` to width results in a width-based reference, with a `width of 1024` and a height of 512.

---

![image](img3.png)

Setting standard to height results in a height-based reference, with a `height of 1024` and a width of 2048.

# Understanding Aspect Ratios

Aspect ratios represent the ratio of width to height. For example, the aspect ratio "16:9" signifies a ratio where the width is 16 and the height is 9.

Simple Interpretation "16:9 = Width:Height"