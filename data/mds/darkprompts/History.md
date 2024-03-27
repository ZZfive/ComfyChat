
v0.0.9 / 2024-03-25
==================

  * Add DarkCheckpointRandomizer and DarkCheckpointSwitcher

v0.0.8 / 2024-03-10
==================

  * New "Randomly Disable" DarkPrompt feature
  * DarkFolder allows dynamically generating folder prefixes based on certain conditions
  * Add DarkFaceIndexGenerator which may be useful when using ReActor
  * Split out into multiple libraries, add new DarkFolders module
  * Fix a crash due to invalid float
  * Fix randomly disabled prompts including combine_with_delimiter

v0.0.7 / 2024-02-03
===================

  * Update workflow image to include DarkFaceIndexShuffle
  * Update README with DarkFaceIndexShuffle information
  * Implement DarkFaceIndexShuffle

v0.0.6 / 2024-01-25
===================

  * Improve debug output to display the chosen line in console output
  * Add DarkLoRALoader for loading LoRAs from tags and emit a LORA_STACK for use with EfficientLoader
  * Fix prompt getting truncated when no data is available in a file or textbox
  * Prefix and suffix should not be returned if there are no lines to chose from (i.e. the node is bypassed)
  * Fix a crash if all lines in a file are commented out and no text input is available
  * Update README now that we are listed in ComfyUI Manager

v0.0.5 / 2024-01-21
===================

  * Update workflow with latest picture and embed workflow data for easy import

v0.0.4 / 2024-01-20
===================

  * Oops--remove debugging print statements

v0.0.3 / 2024-01-20
===================

  * Make Dark Combiner useless by integrating combining into Dark Prompt

v0.0.2 / 2024-01-19
===================

  * Raise an error if a file is not found
  * Forgot to mention the feature that made me write DarkPrompt

v0.0.1 / 2024-01-19
===================

  * Initial release
