# Disclaimer

This is only a fix implemented on top of the original [ComfyUI-nodes-hnmr](https://github.com/hnmr293/ComfyUI-nodes-hnmr) by hnmr293.
The reason why this repo exists is because, hnmr293 has not been approving any pull requests for a while now, and I believe our community deserves to be able to use this amazing tool by hnmr293 without any issues.
All credits go to:
+ [hnmr93](https://github.com/hnmr293)
+ [spacepxl](https://github.com/spacepxl)
+ [WeeBull](https://github.com/WeeBull)

# ComfyUI custom nodes

![cover](./examples/workflow_xyz.png)

## Examples

### X/Y/Z-plot

seeds, steps, cfg scales and others

[workflow link](./examples/workflow_xyz.json)

![xyz plot 1](./examples/workflow_xyz.png)

models

[workflow link](./examples/workflow_xyz_model_clip.json)

![xyz plot 2](./examples/workflow_xyz_model_clip.png)

VAEs

[workflow link](./examples/workflow_xyz_vae.json)

![xyz plot 3](./examples/workflow_xyz_vae.png)

### Merge

simple merge and [merge block weighted](https://github.com/bbc-mc/sdweb-merge-block-weighted-gui) (thanks for @bbc-mc)

#### BMW

[workflow link](./examples/workflow_mbw.json)

![merge block weighted](./examples/workflow_mbw.png)

#### Multi-BMW

BMW with multi-alpha like [supermerger](https://github.com/hako-mikan/sd-webui-supermerger/) (thanks for @hako-mikan)

[workflow link](./examples/workflow_mbw_multi.json)

![multi merge block weighted](./examples/workflow_mbw_multi.png)

### Latent visualization

visualization of 4-channel latent tensor

[workflow link](./examples/workflow_lat2image.json)

![latent visualization](./examples/workflow_lat2image.png)

## Node List

### Latent nodes

|category|node name|input type|output type|desc.|
| --- | --- | --- | --- | --- |
|latent|RandomLatentImage|`INT`, `INT`, `INT`|`LATENT`|(width, height, batch_size)|
|latent|VAEDecodeBatched|`LATENT`, `VAE`, `INT`|`IMAGE`|VAE decoding with specified batch size|
|latent|VAEEncodeBatched|`IMAGE`, `VAE`, `INT`|`LATENT`|VAE encoding with specified batch size|
|latent|LatentToImage|`LATENT`, `FLOAT`, `FLOAT`|`IMAGE`|convert 4-ch latent tensor to 4 grayscale images|
|latent|LatentToHist|`LATENT`, `...`|`IMAGE`|create a histogram of the input latent|

### Sampling nodes

|category|node name|input type|output type|desc.|
| --- | --- | --- | --- | --- |
|sampling|KSamplerSetting|`MODEL`, `CONDITIONING`, `CONDITIONING`, `LATENT`|`DICT`|aggregate sampler's setting to single dict|
|sampling|KSamplerOverrided|various|`LATENT`|override sampler's setting defined by `KSamplerSetting`|
|sampling|KSamplerXYZ|various|`LATENT`|generate latents with values|

### Model nodes and Loader nodes

|category|node name|input type|output type|desc.|
| --- | --- | --- | --- | --- |
|loader|StateDictLoader|(model name)|`DICT`|load state_dict|
|model|Dict2Model|`DICT`, (config_file)|`MODEL`|instantiate a model from given state_dict|
|model|StateDictMerger|`DICT`, `DICT`, `FLOAT`|`MODEL`, `CLIP`, `VAE`|merge two or three models|
|model|StateDictMergerBlockWeighted|`DICT`, `DICT`|`DICT`|merge two models with per-block weights|
|model|StateDictMergerBlockWeightedMulti|`MODEL`, `MODEL`, `STRING`|`MODEL`, `CLIP`, `VAE`|merge two models with per-block weights|
|model|SaveStateDict|`MODEL`, `STRING`, `STRING`|`-`|save state_dict to the output directory|
|model|ModelIter|`MODEL`, `MODEL`|`MODEL`|iterate models|
|model|CLIPlIter|`CLIP`, `CLIP`|`CLIP`|iterate CLIPs|
|model|VAElIter|`VAE`, `VAE`|`VAE`|iterate VAEs|

### Image nodes

|category|node name|input type|output type|desc.|
| --- | --- | --- | --- | --- |
|image|ImageBlend2|`IMAGE`, `IMAGE`, `FLOAT`, `STRING`|`IMAGE`|`ImageBlend` with extra blend modes|
|image|GridImage|`IMAGE`, `INT`, `INT`|`-`|generate single image with specific number of columns|

### Others

|category|node name|input type|output type|desc.|
| --- | --- | --- | --- | --- |
|utils|SaveText|`STRING`, `STRING`, `STRING`|`-`|save texts with specified prefix and ext|
