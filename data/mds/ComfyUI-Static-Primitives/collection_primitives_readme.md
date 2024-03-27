# Adding custom collection primitives

Adding your own collection primitives is quite simple.  
Each group of collection primitives is defined in a `.json` file.  
These files should be placed in this directory:  
`ComfyUI-Static-Primitives/collection_primitives/`

## Collection primitive definition structure

Let's look at an example.
```json
{
    "scheduler": [
        "normal",
        "karras",
        "exponential",
        "sgm_uniform", 
        "simple",
        "ddim_uniform"
        ],
    "sampler" : [
        "euler",
        "euler_ancestral", 
        "heun",
        "heunpp2", 
        "dpm_2",
        "dpm_2_ancestral",
        "lms",
        "dpm_fast",
        "dpm_adaptive",
        "dpmpp_2s_ancestral",
        "dpmpp_sde",
        "dpmpp_sde_gpu",
        "dpmpp_2m",
        "dpmpp_2m_sde",
        "dpmpp_2m_sde_gpu",
        "dpmpp_3m_sde",
        "dpmpp_3m_sde_gpu",
        "ddpm",
        "lcm",
        "ddim",
        "uni_pc",
        "uni_pc_bh2"
    ] 
}
```
This is the contents of the by default included file
`samplers_and_schedulers.json`.  
You can see two key value pairs scheduler and sampler.
The key will be used for the display name of their respective nodes.

The value of each key is a list of strings.
These directly translate to the options for the node's dropdown.

*NOTE*:  
It is important to avoid having duplicate keys across files.
If any duplicate keys are found while loading, all definitions with that key will be ignored