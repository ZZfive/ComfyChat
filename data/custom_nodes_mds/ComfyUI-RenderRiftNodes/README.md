# RenderRift Node Pack

Welcome to the RenderRift Custom Node Pack for the ComfyUI. This was created as I need a couple of extra nodes to help in my Animatediff workflow. Below you'll find detailed explanations of each node and their respective functionalities.

## Nodes Description

### RR_VideoPathMetaExtraction
- **Purpose**: Works in conjunction with VHS_LoadVideoPath to set the file_path for a video and extracts the metadata from the file to send to RR_Image_metadata_overlay.
- **Functionality**:
  - Provides metadata to RR_Image_Metadata_Overlay 
  - Provides file path to VHS_LoadVideoPath

###  RR_Image_Metadata_Overlay
- **Purpose**: Main node for displaying video and metadata in a grid format.
- **Features**:
  - Supports 2-6 videos & their corresponding metadatas.
  - Optional support for an "original video" for comparison in vid2vid workflows.
  - Grid format display with data overlay options.
- **Predefined Options for Overlay**:
  - checkpoint, ksampler, controlnets, animatediff, ipadapter, loras.
  - `search_query`: Enter a class_type from metadata to output all related values.
- **Usage**: Central to creating comparative video overlays with metadata insights.


### RR_Date_Folder_Format Node
- **Purpose**: Outputs a string for saving images/videos in a specific format. To be used as an input to the filename_prefix
- **Output Format**:
  - `{todays_date}/1lq_`
  - `{todays_date}/1_lq/lqimg_`
  - `{todays_date}/1hq_`
  - `{todays_date}/1_hq/hqimg_`
  - `{todays_date}/1facedetailer_`
  - `{todays_date}/1_facedetailer/facedetailerimg_`


## Workflows
[Workflow file here](https://github.com/RenderRift/ComfyUI-RenderRiftNodes/blob/master/.github/workflows/workflow.json)

### Workflow 1: Node Type Listing
![Workflow part 1](.github/images/workflowpt1.png)
- **Description**: Retrieves a list of node types from your video.
- **Steps**:
  - Run the first group in the workflow to obtain the node types.

### Workflow 2: Video Comparison Setup
![Workflow part 2](.github/images/workflowpt2.png)
- **Description**: Sets up video paths for comparison and overlay selection in RR_Image_Metadata_Overlay.
- **Requirements**: Ensure videos have been saved using VHS_VideoCombine with metadata enabled.
- **Steps**:
  - Bypass the first group if node types are known.
  - Select video paths for comparison.
  - Choose overlays for the RR_Image_Metadata_Overlay.
  - Save the output as a comparison video.
