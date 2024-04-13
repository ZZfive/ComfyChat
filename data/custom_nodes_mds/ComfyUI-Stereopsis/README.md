# ComfyUI-Stereopsis

This initiative represents a solo venture dedicated to integrating a stereopsis effect within ComfyUI (Stable Diffusion). By processing a series of sequential images through the Side-by-Side (SBS) node and applying Frame Delay to one of the inputs, it facilitates the creation of a stereopsis effect. This effect is compatible with any Virtual Reality headset that supports SBS video playback, offering a practical application in immersive media experiences.

# Side By Side Module for Image Concatenation
**Overview**

<p align="center">
<img width="1723" alt="Side-by-Side" src="https://github.com/IsItDanOrAi/ComfyUI-Stereopsis/assets/162214102/c4b70e1e-56ff-4010-954c-da2371b87903">
</p>

The Side_by_side module is a utility designed for concatenating two images side by side into a single image. It is implemented in Python using the PyTorch library, making it suitable for both academic research and industrial applications where image manipulation and processing are required.

**Features**

   Image Concatenation: Combines two images horizontally, aligning them side by side.
   Input Flexibility: Accepts two images as input, with no restriction on the images' width, but requires them to have the same height and channel depth.
   PyTorch Integration: Utilizes PyTorch for image manipulation, ensuring compatibility with other PyTorch-based pipelines and operations.

**Technical Details**

   Input Types: The module requires two inputs, each labeled as left_image and right_image. Both inputs are expected to be images, represented as PyTorch tensors.
   Return Types: The output is a single image, which is the result of concatenating the input images side by side.
   Functionality: By leveraging the torch.cat function, it combines the input images along the width dimension, creating a seamless side-by-side image composition.

**Requirements**

   PyTorch: As a PyTorch-based module, it requires an environment where PyTorch is installed and configured.



# Frame Delay Module for Image Sequences
**Overview**

<p align="center">
<img width="677" alt="Frame Delay" src="https://github.com/IsItDanOrAi/ComfyUI-Stereopsis/assets/162214102/a8768fa4-596e-4484-af9c-9bbda21da0f5">
</p>

The FrameDelay module offers a sophisticated solution for manipulating image batches by repeating and inserting frames, thereby introducing a delay effect without altering the overall batch size. This functionality is crucial in applications requiring temporal adjustments within image sequences, such as video processing, animation, and dynamic visual effects creation.
**Features**

   Frame Selection and Delay: Allows the selection of a specific frame within a batch and introduces a delay by repeating the selected frame a specified number of times.
   Batch Size Maintenance: Ingeniously maintains the original batch size after frame insertion, ensuring consistency and compatibility with subsequent processing stages.
   Customizable Parameters: Provides flexibility in selecting the frame to delay and the extent of the delay, with sensible defaults and bounds.

**Technical Details**

   Input Types: Accepts an image batch as input along with parameters to select the frame (selected_frame) and define the delay (frame_delay).
   Return Types: Outputs an image batch where the selected frame has been repeated according to the specified delay, with the batch size kept constant.
   PyTorch Integration: Built with PyTorch, this module seamlessly integrates into workflows that utilize PyTorch for image manipulation and tensor operations.

**Requirements**

   PyTorch: As it relies on PyTorch for all tensor operations, having PyTorch installed and configured in the working environment is a prerequisite.
