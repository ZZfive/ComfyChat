# ComfyUI-SVDResizer
SVDResizer is a helper for resizing the source image, according to the sizes enabled in Stable Video Diffusion.
The rationale behind the possibility of changing the size of the image in steps between the ranges of 576 and 1024, is the use of the greatest common denominator of these two numbers which is 64.
SVD is lenient with resizing that adheres to this rule, so the chance of coherent video that is not the standard size of 576X1024 is greater.
It is advisable to keep the value 1024 constant and play with the second size to maintain the stability of the result.


![SVDResizer demo](https://github.com/ShmuelRonen/ComfyUI-SVDResizer/assets/80190186/3eeccc4d-fcc7-486e-b733-8fa3e985a3c1)
