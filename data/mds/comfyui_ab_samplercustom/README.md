### comfyui_ab_sampler

Experimental sampler node.

Sampling alternates between A and B inputs until only one remains, starting with A.

B steps run over a 2x2 grid, where 3/4's of the grid are copies of the original input latent.

When the optional mask is used, the region outside the defined roi is copied from the
original latent at the end of every step. 