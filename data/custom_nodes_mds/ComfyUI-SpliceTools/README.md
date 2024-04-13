# SpliceTools
Experimental utility nodes with a focus on manipulation of noised latents

## Rerange Sigmas
Given input sigmas, produce a new set of sigmas with the same sigma_max and sigma_min, but reranged to a given number of steps. When combined with split sigmas, it allows for an easy means of changing the amount of steps given to a single sampling node without requiring multiple changes or marking a part of a workflow as stale.

## Splice Latents
Uses a Gaussian blur to approximately split a noised latent into noise and un-noised latents. This can be used to make changes to low frequency detail while preserving higher level detail. For example, it could be used to change hair color while preserving the fine detail of hair strands. Particularly useful with renoise (flip sigmas) workflows

Results are novel, but seem to produce haloing artifacts which are still being investigated, but likely a result of how the frequency filtering is implemented.

## Splice Denoised
When both output and denoised output exist (sampler custom), Splice Denoised allows for a more accurate reproduction of Splice Latents. Note that this does not strip the higher frequency detail of the donor_latent as Splice Latents does with it's lower latent

## Temporal Splice
Performs a splice across the temporal domain, combining the detail which does not move from the lower latent with the detail that does move the upper latent.

This was primarily implemented to experiment with the coloring of line art animations, but has also seen use filtering graphical artifacts introduced by AnimateDiff in rotoscoping workflows.
