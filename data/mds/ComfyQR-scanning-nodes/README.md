# ComfyQR-scanning-nodes

A set of ComfyUI nodes to quickly test generated QR codes for scannability. A companion project to [Comfy-QR](https://gitlab.com/sofuego-comfy-nodes/ComfyQR).

This repository is managed publicly on [Gitlab](https://gitlab.com/sofuego-comfy-nodes/ComfyQR-scanning-nodes), but also mirrored on [Github](https://github.com/coreyryanhanson/ComfyQR-scanning-nodes). Please submit any [issues](https://gitlab.com/sofuego-comfy-nodes/ComfyQR-scanning-nodes/-/issues) or [pull requests](https://gitlab.com/sofuego-comfy-nodes/ComfyQR-scanning-nodes/-/merge_requests) to the gitlab repo.

## Getting started

This project currently contains two custom nodes. One for extracting text from an image that contains a QR code and another to validate whether that text was readable and matches the link on your QR code.

**Important:** QR reading uses the pyzbar module, which requires the zbar library to be installed. **_It will not work without zbar installed to your system._** Each OS has a different way of doing so and more detailed instructions can be found in the [pyzbar repository](https://github.com/NaturalHistoryMuseum/pyzbar/).

### Read QR Code

A node that extracts the text data from a QR code using the pyzbar library.

#### Inputs

* `image` - A piped input for the image layer.
* `library` - The QR reader library. Currently has only `pyzbar` available, but more may be added in future updates if the benefits outweigh the additional dependencies.

#### Outputs

* `EXTRACTED_TEXT` - A string of text extracted from the QR code. If extraction could not be performed it will output an empty string.

### Validate QR Code

A node that allows you to check whether text is present and whether it is matching (optionally allowing the user to either interrupt the process if the test fails or pass the check through with a return code that could be used in other custom nodes for more advanced applications.)

#### Inputs

* `image` - A piped input for the image layer.
* `extracted_text` A piped input for text data coming from the `Read QR Code` node.
* `protocol` - If enabled this will prefix the textbox input with a preset to represent the internet protocol. This is included both for convenience and as a workaround for the textbox clipping strings with this character combination.
  * `Http` - Adds "http://" before the text.
  * `Https` - Adds "https://" before the text.
  * `None` - Uses only the contents of the `text` box.
* `text` - The text from the QR code before any AI processing. This (combined with `protocol`) will be compared against the `extracted_text`.
* `passthrough` - If set to `False` pipelines will be interrupted when a QR fails the readability and match tests. When set to `True`, it will be bypassed.


#### Outputs

* `IMAGE` - The original image layer passed through. Ensures that failed attempts can be stopped before reaching a `Save Image` node.
* `VALIDATION_CODE` - An integer for a custom return code of the QR check. `0` indicates a perfect match, `1` indicates an unreadable QR, and `2` indicates a text mismatch.

## Accuracy in readability

Different QR libraries each have their indivual pros and cons. Currently `pyzbar` is chosen based on amount of dependencies, recency of updates, and popularity. It is not perfect and different libraries will sometimes disagree on QR readability.

To reduce false negatives with `pyzbar`, I noticed that when the border size of a QR is small, it may help to add a `Pad Image for Outpainting` node before sending it to `Read QR Code`.
