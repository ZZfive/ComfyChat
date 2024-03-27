# ComfyUI-Advanced-Latent-Control

**This custom nodes helps to transform latent in different ways.**

>You can access new features earlier by switching from the master branch to dev,
but you need to remember that there may be some issues on the dev branch and some nodes' behavior may change after release.

## Latent mirror
This node can flip latent and merge original and flipped version.

**Input:** 
- `latent`

**Fields:**
- `direction` – can be `vertically`, `horizontally` or `both`
- `multiplier` – multiply latent by specified number

**Output:**
- `latent`

**Usage:**  
![sample](https://i.imgur.com/YMyYorQ.png)
![sample](https://i.imgur.com/W5BasCO.png)

## Latent shift
This node can shift latent along x and y-axis.

**Input:** 
- `latent`

**Fields:**
- `x_shift` – a number between -1 and 1 that indicates how much the latent should be shifted
- `y_shift` – a number between -1 and 1 that indicates how much the latent should be shifted

**Output:**
- `latent`

**Usage:**  
![sample](https://i.imgur.com/1Dp5dSw.png)

## TSampler with transforms (Latent Control)
This node can multiply, mirror and shift latent during generation.

**Input:**  
exactly matches the base KSampler

**Fields:**
- base KSampler fields
- `start_mirror_at` – a number between 0 and 1 that indicates at what point the sampler will start mirroring
- `stop_mirror_at` – a number between 0 and 1 that indicates at what point the sampler will stop mirroring
- `mirror_mode` – can be `replace` or `combine`. `replace` will replace the latent with the transformed one, `combine` will add the original and the transformed latent and divide by 2
- `mirror_direction` – can be `none`, `vertically`, `horizontally`, `both`, `90 degree rotation` or `180 degree rotation`
- `start_shift_at` – a number between 0 and 1 that indicates at what point the sampler will start shifting
- `stop_shift_at` – a number between 0 and 1 that indicates at what point the sampler will stop shifting
- `shift_mode` – can be `replace` or `combine`. `replace` will replace the latent with the transformed one, `combine` will add the original and the transformed latent and divide by 2
- `x_shift` – a number between -1 and 1 that indicates how much the latent should be shifted
- `y_shift` – a number between -1 and 1 that indicates how much the latent should be shifted
- `start_multiplier_at` – a number between 0 and 1 that indicates at what point the sampler will start multiplying
- `stop_multiplier_at` – a number between 0 and 1 that indicates at what point the sampler will stop multiplying
- `multiplier_mode` – can be `replace` or `combine`. `replace` will replace the latent with the transformed one, `combine` will add the original and the transformed latent
- `multiplier` – multiply latent by specified number

**Output:**
exactly matches the base KSampler

**Usage:**  
**You also can use those params together**  
![sample](https://i.imgur.com/RMJTnWF.png)  
![sample](https://i.imgur.com/fQ7UWuS.png)  
![sample](https://i.imgur.com/pxWupAx.png)  
![sample](https://i.imgur.com/1YkERDu.png)  

## TSampler (Latent Control)
This node allows to combine a lot of transforms with different parameters.

**Input:**
- base KSampler fields
- transform_optional – field that can take output from one of those nodes: `Mirror transform`, `Shift transform`, `Multiply transform` or `Combine transforms`

**Fields:**
exactly matches the base KSampler

**Output:**
exactly matches the base KSampler

**Usage:**  
![sample](https://i.imgur.com/PlGnAtA.png)  
![sample](https://i.imgur.com/CtrBRPn.png)  

Multiply, Mirror and Shift transform nodes parameters exactly match the corresponding `KSampler with transforms (Latent Control)` parameters.

There are two new transform nodes:
- Latent add
- Latent interpolate

They work exactly the same as LatentAdd and LatentBlend nodes from standard node pack, but also, can multiply result by specified number.

## Offset
You can apply specific offset for transform nodes.

**Fields:**
- `process_every` – a number that indicates which steps will be processed
- `offset` – a number that indicates offset for previous parameter. For example: if `process_every` is 4 and `offset` is 0, sampler apply transformation with this pattern: **0 0 0 1**. This pattern will repeat again and again. If `offset` is 2, pattern will be **0 1 0 0**, if -1 – **1 0 0 0**.
- `mode` – can be `process_every` or `skip_every`. For example, with `skip_every` previous pattern (**0 0 0 1**) turn into this: **1 1 1 0**

**Output:**
- `offset`

**Usage:**  
![sample](https://i.imgur.com/ExZacqG.png)
![sample](https://i.imgur.com/tR6KSmI.png)
![sample](https://i.imgur.com/MGVLfve.png)

You can combine different offsets to achieve interesting patterns. For example:  
**0 0 0 1** and **0 0 1** give this pattern: **0 0 1 1 0 1 0 1 1 0 0 1**.

## One time nodes

Each transform node has own one-time version. They allow to make one transform action at specified step.

**Usage:**  
![sample](https://i.imgur.com/Q1Vyob0.png)

## Latent normalize

Fixes some issues when sampling modified latent space.

**Input**  
exactly matches the `VAE Decode` node

**Output**
- latent

When you multiply latent by negative or big positive (bigger than 2) number and paste this latent in sampler, you can see that the
image will be generated very poorly. This is because stable diffusion cannot work with such set of numbers (meaning the numbers contained in latent).

![sample](https://i.imgur.com/3FXk8n7.png)

But you can prevent this behavior by sequential decode and encode latent using vae. Node `Latent normalize` make this process easier.

![sample](https://i.imgur.com/hkFYYVh.png)

This node also change some results even if output without this node looks good.

![sample](https://i.imgur.com/kP0f6vh.png)
![sample](https://i.imgur.com/YI8ZqLd.png)

And it very slightly changes results from latent, which have not been modified.

![sample](https://i.imgur.com/xTU08xm.png)
![sample](https://i.imgur.com/yzgW7QT.png)


