## Lists Cartesian Product ( [ComfyUI](https://github.com/comfyanonymous/ComfyUI) custom node )

Given a set of lists, the node adjusts them so that when used as input to another node all the possible argument permutations are computed.

To update the input and output slots, **right-click the node and at the top select the option "update I/Os"**.


### Example

Given the following lists as inputs:

- ```Positive_Cond_List_: [ A, B, C]```
- ```Negative_Cond_List_: [ a, b ]```
- ```__________CFG_List_: [ 5, 7 ]```

The outputs are the following:

- ```Positive_Cond_List_: [ A, A, A, A, B, B, B, B, C, C, C, C]```
- ```Negative_Cond_List_: [ a, a, b, b, a, a, b, b, a, a, b, b]```
- ```__________CFG_List_: [ 5, 7, 5, 7, 5, 7, 5, 7, 5, 7, 5, 7]```


### Example Workflow 

*image has workflow embedded*

![example workflow](/workflows/example.png)