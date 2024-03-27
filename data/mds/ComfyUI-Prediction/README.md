# ComfyUI-Prediction
Fully customizable Classifier Free Guidance for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

![Avoid and Erase Workflow](examples/avoid_and_erase.png)

Copyright 2024 by @RedHotTensors and released by [Project RedRocket](https://huggingface.co/RedRocket).

# Installation
Clone this repo into ``ComfyUI/custom_nodes`` or use [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager).

(Optional) If you want beautiful teal PREDICTION edges like the example apply [patches/colorPalette.js.patch](https://raw.githubusercontent.com/redhottensors/ComfyUI-Prediction/main/patches/colorPalette.js.patch) to ``ComfyUI/web/extensions/core/colorPalette.js``.

# Usage
All custom nodes are provided under <ins>Add Node > sampling > prediction</ins>. An example workflow is in ``examples/avoid_and_erase.json``.

Follow these steps for fully custom prediction:
1. You will need to use the <ins>sampling > prediction > Sample Predictions</ins> node as your sampler.
2. The *sampler* input comes from <ins>sampling > custom_sampling > samplers</ins>. Generally you'll use **KSamplerSelect**.
3. The *sigmas* input comes from <ins>sampling > custom_sampling > schedulers</ins>. If you don't know what sigmas you are using, try **BasicScheduler**. (NOTE: These nodes are **not** in the "sigmas" menu.)
4. You'll need one or more prompts. Chain <ins>conditioning > CLIP Text Encode (Prompt)</ins> to <ins>sampling > prediction > Conditioned Prediction</ins> to get started.
5. After your prediction chain, connect the result to the *noise_prediction* input of your **Sample Predictions** node.

# Predictors

## Primitive Nodes
All other predictions can be implemented in terms of these nodes. However, it may get a little messy.

**Conditioned Prediction** - Evaluate your chosen model with a prompt (conditioning). You need to pick a unique conditioning name like "positive", "negative", or "empty". (The names are arbitrary and you can choose any name, but the names may evenutally interact with ControlNet if/when it's implemented.)

**Combine Predictions** - Operates on two predictions. Supports add (+), subtract (-), multiply (*), divide (/), [vector projection](https://en.wikipedia.org/wiki/Vector_projection) (proj), [vector rejection](https://en.wikipedia.org/wiki/Vector_projection) (oproj), min, and max.<br>
``prediction_A <operation> prediction_B``

**Scale Prediction** - Linearly scales a prediction.<br>
``prediction * scale``

**Scaled Guidance Prediction** - Combines a baseline prediction with a scaled guidance prediction using optional standard deviation rescaling, similar to CFG.<br>
Without stddev_rescale: ``baseline + guidance * scale``<br>
With stddev_rescale: [See §3.4 of this paper.](https://arxiv.org/pdf/2305.08891.pdf) As usual, start out around 0.7 and tune from there.

## Convinence Nodes

**Avoid and Erase Prediction** - Re-aligns a desirable (positive) prediction called *guidance* away from an undesirable (negative) prediction called *avoid_and_erase*, and erases some of the negative prediction as well.<br>
``guidance - (guidance proj avoid_and_erase) * avoid_scale - avoid_and_erase * erase_scale``

## Prebuilt Nodes
**CFG Prediction** - Vanilla Classifer Free Guidance (CFG) with a postive prompt and a negative/empty prompt. Does not support CFG rescale.<br>
``(positive - negative) * cfg_scale + negative``

**Perp-Neg Prediction** - Implements https://arxiv.org/abs/2304.04968. (The built-in ComfyUI Perp-Neg node is [incorrectly implemented](https://github.com/comfyanonymous/ComfyUI/issues/2858).)<br>
``pos_ind = positive - empty; neg_ind = negative - empty``<br>
``(pos_ind - (neg_ind oproj pos_ind) * neg_scale) * cfg_scale + empty``

# Limitations
ControlNet is not supported at this time.

Regional prompting may work but is totally untested.

Any other advanced features affecting conditioning are not likely to work.

# License
The license is the same as ComfyUI, GPL 3.0.
