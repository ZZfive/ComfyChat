## ImageTextOverlay Node for ComfyUI
ImageTextOverlay is a customizable Node for ComfyUI that allows users to easily add text overlays to images within their ComfyUI projects. This Node leverages Python Imaging Library (PIL) and PyTorch to dynamically render text on images, supporting a wide range of customization options including font size, alignment, color, and padding.

## Features
- Dynamic text overlay on images with support for multi-line text.
- Automatic text wrapping and font size adjustment to fit within specified dimensions.
- Customizable text alignment (left, right, center), color, and padding.
- Easy integration into ComfyUI workflows.

## Installation
This Node is designed to be used within ComfyUI. Ensure you have ComfyUI installed and running in your environment. Use git clone https://github.com/Big-Idea-Technology/ImageTextOverlay to custom_nodes in ComfyUI folder. For details on installing ComfyUI, refer to the official documentation.

## Usage
To use the ImageTextOverlay Node in your ComfyUI project, follow these steps:

Add the Node to Your Project: Ensure the ImageTextOverlay class is properly integrated into your ComfyUI environment. The class should be located in a custom_nodes that's accessible by your ComfyUI project.

## Configure the Node: 
Within your ComfyUI project, configure the ImageTextOverlay Node with the necessary parameters:

- image: The image to overlay text onto.
- text: The text to overlay.
- textbox_width & textbox_height: The dimensions of the text box.
- max_font_size: The maximum font size to use.
- font: Path to the font file.
- alignment: The alignment of the text within the box (left, right, center).
- color: The color of the text.
- start_x & start_y: The starting position of the text box on the image.
- padding: Padding around the text inside the text box.

## Contributing
Contributions to improve the ImageTextOverlay Node or add new features are welcome! Please follow the project's contribution guidelines for submitting issues or pull requests.

## License
MIT License - Feel free to use and modify the ImageTextOverlay Node for your personal or commercial projects.

## Credit
Credits to https://github.com/Smuzzies/comfyui_chatbox_overlay 