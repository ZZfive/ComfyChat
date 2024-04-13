
# Pooling modes comparison

## SSD-1B based model with lcm-lora

 - checkoint: [ssd-1b-animagine](https://huggingface.co/furusu/SSD-1B-anime)
 - lcm-lora: [lcm-animagine](https://huggingface.co/furusu/SD-LoRA/blob/main/lcm-animagine.safetensors)
 - Sampler: lcm
 - Scheduler: normal
 - Steps: 6
 - cfg: 2.2
 - seed: 1

| Without | Initial parameters (blend=0) | Tuned paramters |
|:-------:|:------------------:|:---------------:|
| ![](images/ssd1b_lcm.webp) | ![](images/ssd1b_lcm_sa.webp) | ![](images/ssd1b_lcm_mix.webp) |


mode comparison

| mode | 1D | 2D |
|:----:|:--:|:--:|
| AVG  | ![](images/ssd1b_lcm_avg1d.webp) | ![](images/ssd1b_lcm_avg2d.webp) |
| MAX  | ![](images/ssd1b_lcm_max1d.webp) | ![](images/ssd1b_lcm_max2d.webp) |



## SD 1.5 based models

 - checkoint: [Blazing Drive v10g](https://civitai.com/models/121083?modelVersionId=167841)
 - Sampler: DDIM
 - Scheduler: Karras
 - Steps: 16
 - cfg: 5.0
 - seed: 1

| Without | Initial parameters (blend=0) | Tuned paramters |
|:-------:|:------------------:|:---------------:|
| ![](images/sd15_ddim.webp) | ![](images/sd15_ddim_sa.webp) | ![](images/sd15_ddim_mix.webp) |

mode comparison

| mode | 1D | 2D |
|:----:|:--:|:--:|
| AVG  | ![](images/sd15_ddim_avg1d.webp) | ![](images/sd15_ddim_avg2d.webp) |
| MAX  | ![](images/sd15_ddim_max1d.webp) | ![](images/sd15_ddim_max2d.webp) |
