
# Tiling sizes comparison

## SD 1.5 based models

 - checkoint: [Blazing Drive v10g](https://civitai.com/models/121083?modelVersionId=167841)
 - Sampler: euler_ancestral
 - Scheduler: normal
 - Steps: 12
 - cfg: 5.0
 - seed: 1
 - image size: 1024x1536 (sample images are shrinked to 512x768)

| global_ratio | tile_size 48 | tile_size 32 | tile_size 24 |
|:-------:|:------------:|:------------:|:------------:|
| 0.0 | ![](images/sd15_nst_0_48.webp) | ![](images/sd15_nst_0_32.webp) | ![](images/sd15_nst_0_24.webp) |
| 0.5 | ![](images/sd15_nst_5_48.webp) | ![](images/sd15_nst_5_32.webp) | ![](images/sd15_nst_5_24.webp) |
| 1.0 | ![](images/sd15_nst_10_48.webp) | ![](images/sd15_nst_10_32.webp) | ![](images/sd15_nst_10_24.webp) |
