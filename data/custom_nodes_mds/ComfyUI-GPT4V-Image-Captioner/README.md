# ComfyUI-GPT4V-Image-Captioner
Fill in the key and URL to quickly call GPT4V to annotate images

### Input Image Processing:
The project automatically handles image processing, eliminating the need for manual scaling.

### GPT4V Integration:
To utilize GPT4V for image annotation, you need to provide the API key and API URL.

### Seed and OpenAI Labeling:
The seed value determines the consistency of the labeling process. If the seed and image remain unchanged, there will be no additional calls to OpenAI for labeling.
If you are not satisfied with the labeling results, you can modify the seed value to get different outputs.

### Prompt Types:
There are two prompt types available: "generic" and "figure".
Under the "figure" type, labels will exclude color terms, backgrounds, watermarks, signatures, etc., focusing only on the overall characteristics of the characters.

### Weighted Labels:
The option to enable weights for labels is available.
When weights are enabled, corresponding weight values will be assigned to the labels.

### Excluding Unwanted Words:
If there are specific words in the labels that you want to exclude, you can enter them in the "exclude_words" field.
The system will automatically exclude the specified words from the generated labels.

