# Amazon Bedrock nodes for ComfyUI

[__*Amazon Bedrock*__](https://aws.amazon.com/bedrock/) is a fully managed service that offers a choice of high-performing foundation models (FMs) from leading AI companies.
This repo is the ComfyUI nodes for Bedrock service. You can invoke foundation models in your ComfyUI pipeline.

## Installation (SageMaker by CloudFormation)

Using [__*Amazon SageMaker*__](https://aws.amazon.com/sagemaker/) is the easiest way to develop your AI model. You can deploy a ComfyUI on SageMaker notebook using CloudFormation.

1. Open [CloudFormation console](https://console.aws.amazon.com/cloudformation/home#/stacks/create), and upload [`./assets/comfyui_on_sagemaker.yaml`](https://raw.githubusercontent.com/yytdfc/ComfyUI-Bedrock/main/assets/comfyui_on_sagemaker.yaml) by "Upload a template file".

2. Next enter a stack name, choose a instance type fits for you.  Just next and next and submit.

3. Wait for a moment, and you will find the ComfyUI url is ready for you. Enjoy!

![](./assets/stack_complete.webp)


## Installation (Manually)
1. Clone this repository to your ComfyUI `custom_nodes` directory:

``` bash
pip install -r requirements.txt
cd ComfyUI/custom_nodes
git clone https://github.com/yytdfc/ComfyUI-Bedrock.git

# better to work with some third-party nodes
git clone https://github.com/WASasquatch/was-node-suite-comfyui.git
git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git
```

2. You need to make sure your access to Bedrock models are granted. Go to aws console [*https://.console.aws.amazon.com/bedrock/home#/modelaccess*](https://console.aws.amazon.com/bedrock/home#/modelaccess) . Make sure these models in the figure are checked.

![](./assets/model_access.webp)


3. You need configure credential for your environments with IAM Role or AKSK.

- IAM Role

If you are runing ComfyUI on your aws instance, you could use IAM role to control the policy to access to Bedrock service without AKSK configuration.

Open the IAM role console of your running instance, and attach `AmazonBedrockFullAccess` policy to your role.

Alternatively, you can create an inline policy to your role like this:

``` json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": "bedrock:*",
            "Resource": "*"
        }
    ]
}
```


- AKSK (AccessKeySecretKey)

You need to make sure the AKSK user has same policy as the IAM role described before. You can use the aws command tool to configure your credentials file:

```
aws configure
```

Alternatively, you can create the credentials file yourself. By default, its location is ~/.aws/credentials. At a minimum, the credentials file should specify the access key and secret access key. In this example, the key and secret key for the account are specified in the default profile:

```
[default]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
```

You may also want to add a default region to the AWS configuration file, which is located by default at ~/.aws/config:

```
[default]
region=us-east-1
```

If you haven't set the default region and running on aws instance, this nodes will automatically use the same region as the running instance.


## Example

For example, you can use the Bedrock LLM to refine the prompt input and get a better result. Here is an example of doing prompt translation and refinement, and the invoke the image generation model (eg. SDXL, Titan Image) provided by Bedrock.
The result is much better after preprocessing of prompt compared to the original SDXL model (the bottom output in figure) which doesn't have the capability of understanding Chinese. Workflow examples are in `./workflows`.

![](./assets/example.webp)



## Support models:

Here are models ready for use, more models are coming soon.

- Anthropic:
    - [x] Claude (1.x, 2.0, 2.1)
    - [x] Claude Instant (1.x)

- Amazon:
    - Titan Image Generator G1 (1.x)
        - [x] text to image
        - [ ] inpainting
        - [ ] outpainting
        - [ ] image variation


- Stability AI:
    - Stable Diffusion XL (1.0)
        - [x] text to image
        - [ ] image to image
        - [ ] image to image (masking)
