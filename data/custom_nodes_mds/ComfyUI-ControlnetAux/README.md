# ComfyUI Custom Node: ControlNet Auxiliar

This ComfyUI custom node, ControlNet Auxiliar, provides auxiliary functionalities for image processing tasks. It supports various image manipulation and enhancement operations.

# Installation

Before using any source code from external libraries or services, it's advisable to test it in a safe environment such as sandbox machines or remote online services like Google Colab. These platforms provide isolated environments where you can evaluate the code without risking your local setup. Testing in such environments allows you to assess compatibility, functionality, and potential dependencies before integrating the code into your own projects. This practice helps in identifying any issues or conflicts early on and ensures a smoother integration process into your development workflow.

You can find an example of testing ComfyUI with my custom node on `Google Colab` in this [ComfyUI Colab notebook](/notebooks/ComfyUI-ControlnetAux.ipynb).

### Python Package Requirements

- **timm==0.6.12**
- **controlnet-aux==0.0.7**
- **mediapipe**

# Manual Installation Guide for ComfyUI-ControlnetAux (Ubuntu Linux).  

1. **Clone the Repository:**
   - Open your terminal.
   - Navigate to the directory where ComfyUI is installed, ensure that you are in the `custom_nodes` folder (e.g., `ComfyUI/custom_nodes/` ).
   - Run the following command:
     ```bash
     git clone https://github.com/madtunebk/ComfyUI-ControlnetAux.git
     ```

2. **Install Dependencies:**
   - Navigate into the cloned directory:
     ```bash
     cd ComfyUI-ControlnetAux
     ```
   - Activate the virtual environment:
     ```bash
     source ../../venv/bin/activate
     ```
   - Install the required Python packages using pip:
     ```bash
     pip install -r requirements.txt
     ```
   - To support **DWPose** which is dependent on **MMDetection**, **MMCV**, and **MMPose**, you need to install additional packages:
     ```bash
     pip install openmim
     mim install mmengine
     mim install "mmcv>=2.0.1"
     mim install "mmdet>=3.1.0"
     mim install "mmpose>=1.1.0"
     ```

3. **Run the Application:**
   - Once the dependencies are installed, you can run the application.


## Usage

### Input Types

- **image**: Input image for processing.
- **mode**: Processing mode, including options like scribble_hed, softedge_hed, depth_midas, openpose, etc.

### Return Types

The node returns processed images.

### Functionality

- **process_image**: Processes input image based on specified mode and parameters.

### Working Resolution
The default working resolutions for ControlNet models in this repository are `512x512`, `512x768`, `768x512`, `768x768`, `768x1024`, `1024x768`, and `1024x1024` pixels. These resolutions are recommended for optimal performance and output quality.

### Custom Resolutions
Please note that using non-default resolutions may result in variations in output size. While the models are designed to handle a range of input resolutions, output sizes may not always be guaranteed to match the input resolution, especially for non-square or non-standard resolutions.

### Warning
It's important to be aware that deviations from the default working resolution may affect the quality and consistency of the output. If you choose to use custom resolutions, we recommend testing and validating the output to ensure it meets your requirements.

## Example Workflow
![Example Workflow](/workflows/example.png) [Download Example Workflow JSON](/workflows/example.json)

## Note: Work in Progress

This project is under development. Contributions and feedback are welcome!

## Node Class Mappings

The ControlNet Auxiliar node is mapped to various classes corresponding to different models:

- `controlaux_hed`: HED model for edge detection.
- `controlaux_midas`: Midas model for depth estimation.
- `controlaux_mlsd`: MLSD model for line segment detection.
- `controlaux_openpose`: Openpose model for human pose estimation.
- `controlaux_dwpose`: DWPose model for human pose estimation.
- `controlaux_pidi`: PidiNet model for image inpainting.
- `controlaux_normal_bae`: NormalBae model for image restoration.
- `controlaux_lineart`: Lineart model for image stylization.
- `controlaux_lineart_anime`: Lineart Anime model for anime-style image stylization.
- `controlaux_zoe`: Zoe model for depth super-resolution.
- `controlaux_sam`: SAM model for image segmentation.
- `controlaux_leres`: Leres model for image restoration.
- `controlaux_canny`: Canny model for edge detection.
- `controlaux_content`: Content Shuffle model for image processing.
- `controlaux_face_detector`: Mediapipe Face Detector for face detection.

---

**Credits**: Developed using source code from [controlnet_aux](https://github.com/huggingface/controlnet_aux) by [Hugging Face](https://github.com/huggingface). Pretrained models provided by [lllyasviel](https://github.com/lllyasviel), [wanghaofan](https://huggingface.co/wanghaofan), [ybelkada](https://huggingface.co/ybelkada), and [openmmlab](https://openmmlab.com/). 

