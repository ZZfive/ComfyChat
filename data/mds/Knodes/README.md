ComfyUI Nodes

3 Nodes:

- Image(s) to Websocket (Base64)
-- Accepts a batch of image tensors and returns an array of base64 encoded images using the websocket. Returns a string (Actions) for routing.

- Load Image (Base64)
-- Accepts a base64 encoded image and returns an image tensor and mask.

- Load Images (Base64)
-- Accepts a string with the following structure: 0x4(Image count) 0x8(Image1 length) Image1(base64) ... 0x8(ImageN length) ImageN(base64).
-- Returns a batch of image tensors and masks.
