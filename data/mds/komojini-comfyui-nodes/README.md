# komojini-comfyui-nodes
Custom ComfyUI Nodes for video generation

- [ DragNUWA Image Canvas](#dragnuwaimagecanvas)
- [ Flow Nodes](#flownodes)
- [Getter & Setter Nodes](#gettersetternodes)
- [ Video Loading Nodes](#videoloadingnodes)
  - [ Ultimate Video Loader](#ultimatevideoloader)
  - [ YouTube Video Loader](#youtubevideoloader)


<a name="dragnuwaimagecanvas"></a>
## DragNUWA Image Canvas
![drag_preview](https://github.com/komojini/komojini-comfyui-nodes/assets/118584718/82d534c3-df36-46d1-ab16-ff9d3d166e5a)
![7998A098-B97B-45F3-9CE2-3591AC3FFAB4](https://github.com/komojini/komojini-comfyui-nodes/assets/118584718/6ab3b14a-8995-4f11-8966-10740fc9eceb)

Used for DragNUWA nodes witch is from: [https://github.com/chaojie/ComfyUI-DragNUWA](https://github.com/chaojie/ComfyUI-DragNUWA)

DragNUWA main repo: https://github.com/ProjectNUWA/DragNUWA


![741E724B-E861-4C93-9E38-D61B06FFD14D_4_5005_c](https://github.com/komojini/komojini-comfyui-nodes/assets/118584718/089ea987-9a6d-4868-bad4-f7f44e2bc85b)

<a name="flownodes"></a>
## Flow Nodes
Flow node that ables to run only a part of the entire workflow.
By using this, you will be able to generate images or videos "step by step"
Add the "FlowBuilder" node right before the output node (PreviewImage, SaveImage, VideoCombine, etc.), then it will automatically parse only the nodes for generating that output.

### FlowBuilder
![image](https://github.com/komojini/komojini-comfyui-nodes/assets/118584718/97d7e0f0-7ed2-44af-929a-35e6cf3aa622)

### FlowBuilderSetter

### (advanced) Flowbuilder Nodes

<a name="gettersetternodes"></a>
## Getter & Setter Nodes
![image](https://github.com/komojini/komojini-comfyui-nodes/assets/118584718/a01be34e-f8df-4e6f-9364-d9b26de1a097)

Getter & Setter nodes that ensures execution order by connecting them when starting the prompt.

<a name="videoloadingnodes"></a>
## Video Loading Nodes
<a name="ultimatevideoloader"></a>
### Ultimate Video Loader
Able to load video from several sources (filepath, YouTube, etc.)<br>
3 source types available: 
- file path
- file upload
- youtube
- empty video
<br><br>
![image](https://github.com/komojini/komojini-comfyui-nodes/assets/118584718/c2c27476-45e8-462f-a714-3150df1bb633)


Common Args:
- start_sec: float
- end_sec: float (0.0 -> end of the video)
- max_fps: int (0 or -1 to disable)
- force_size
- frame_load_cap: max frames to be returned, the fps will be automatically changed by the duration and frame count. This will not increase the frame count of the original video (will not increase original fps).
<br>
The video downloaded from YouTube will be saved in "path-to-comfyui/output/youtube/" (will be changed later)
<br>

### Ultimate Video Loader (simple)
Same as above but without preview.
<br><br>
<a name=youtubevideoloader></a>
### YouTube Video Loader
<img width="50%" alt="Youtube video loader" src="https://github.com/komojini/komojini-comfyui-nodes/assets/118584718/65142191-f7e9-4341-ba47-4226b31451fd"><br>
Able to load and extract video from youtube.

Args:
- Common Args Above...
- output_dir (optional): defaults to "path-to-comfyui/output/youtube/"

## Others
### Image Merger
Able to merge 2 images or videos side by side.
Useful to see the results of img2img or vid2vid.

divide_points: 2 points that creates a line to be splitted.
One point will be like (x, y) and the points should be seperated by ";".
for "x" and "y", you can use int (pixel) or with %.
e.g. 
- (50%, 0);(50%, 100%) -> split by vertical line in the center
- (0%, 50%);(100%, 50%) -> split by horizontal line in the center
- (40%, 0);(70%, 100%) ->

<img width="80%" src="https://github.com/komojini/komojini-comfyui-nodes/assets/118584718/8839b1da-e5c1-41a9-87e4-514e25e113b5"/>

<img width="80%" src="https://github.com/komojini/komojini-comfyui-nodes/assets/118584718/585b46d7-2a73-4cc2-be29-68d02db0fe1c"/>

<a name="statusviewer"></a>
## System Current Status Viewer
Shows current status of GPU, CPU, and Memory every 1 second.

<p float="left">
  <img src="https://github.com/komojini/komojini-comfyui-nodes/assets/118584718/64954343-d75f-4510-8664-1fafdd40a83d" height="400" />
  <img src="https://github.com/komojini/komojini-comfyui-nodes/assets/118584718/5d61b82f-58a8-4309-a58d-8bdc764adcb9" height="400" />
</p>

- Current GPU memory, usage percentage, temperature
- Current CPU usage
- Current RAM usage

Go to settings and check "🔥 Show System Status" to enable it.

