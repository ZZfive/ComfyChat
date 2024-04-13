# ComfyUI_API_Manager
```
# Custom Workflow Nodes for API Integration

This package provides three custom nodes designed to streamline workflows involving API requests, dynamic text manipulation based on API responses, and image posting to APIs. These nodes are particularly useful for automating interactions with APIs, enhancing text-based workflows with dynamic data, and facilitating image uploads.

## Installation

To use these custom nodes, clone this repository into your `custom_nodes` directory. No external dependencies are required.

```bash
git clone https://github.com/CC-BryanOttho/ComfyUI_API_Manager.git custom_nodes
```

## Usage

### API Request Node

This node performs API requests and processes the responses:

- **auth_url**: Specify the authentication endpoint of the API.
- **auth_body_text**: Enter the payload for the API authentication request.
- **array_path**: Navigate through the response object to locate the desired object or array, especially useful if it's nested.

1. Configure the `auth_url` and `auth_body_text` to authenticate with your target API.
2. Use the `array_path` to specify the path to the data of interest in the API response.

### Text Prompt Combiner Node

Utilizes API response data for dynamic text manipulation:

- Dynamically replace text in a template using `$attributeName` syntax to insert values from the API response.
- **id_field_name**: Specify the attribute name whose value is returned as a string, aiding in identifying or referencing objects.

1. Insert API response data into a text box, using `$attribute1.attribute2` to reference nested attributes.
2. The `id_field_name` can be used to extract specific values from the response for further use.

### Image Post Node

Facilitates the posting of images to an API:

- Requires an image, an object ID as a string, and an API key for authentication.
- **api_url**: The endpoint for image posting, with `$id` used in the URL to dynamically insert the ID.

1. Provide the necessary image, API object ID, and API key.
2. Configure the `api_url`, ensuring to include `$id` where the object ID should be inserted in the URL.

## Example Workflow

Imagine a workflow where you need to fetch data from an API, use part of that data to generate a text prompt, and then post an image related to that prompt to another API endpoint:

1. **API Request Node** fetches the desired data.
2. **Text Prompt Combiner Node** generates a dynamic text prompt based on the fetched data.
3. **Image Post Node** posts an image using details from the previous nodes.

## Additional Information

These nodes are designed to be flexible and can be adjusted or extended based on specific workflow requirements. For more complex API interactions or additional dynamic capabilities, consider customizing these nodes further.

## Contributing

Contributions are welcome! If you have improvements or bug fixes, please submit a pull request or open an issue.

```
