# ComfyUI Path Helper

This module provides a set of custom nodes to help with creating file paths for ComfyUI. Paths are joined using python os library. This should help issues where people are manually constructing paths with strings which then dont work for someone if they are using a different OS then them

## Features

- **Create project root folder**
- **Add Folders**
- **Add file prefix**
- **Show path/strings in a node**
- **Prefix/Postfix datetime to paths**
- **Format variables into paths**

## Nodes
#### CreateProjectRoot
#### AddFolder
#### AddFolderAdvanced
#### AddFileNamePrefix
#### AddFileNamePrefixAdvanced
#### JoinVariables
#### ShowPath
#### ShowString

## Basic usage
![basic_useage.png](imgs%2Fbasic_useage.png)
## Advanced usage

### Date Time
Advanced Nodes allow you to prefix or postfix datetime to a path
Datetime is formatted using strftime so can be customized to any format you like
more info on strftime here https://strftime.org/
![adv_datetime_use.png](imgs%2Fadv_datetime_use.png)

### Variable formatting
Advanced Nodes also allow for the formatting of variables with the node being constructed. Use the {} notion where you wish for variable to be formatted within the given string
![variable_formatting.png](imgs%2Fvariable_formatting.png)

### Multiple Variable formatting
If you wish to format multiple variables then this is also possible by passing a comma seperated string into 'input_variables'. You can use the Join Variables node to do this for you.
Note order of variables is import
![multi_variable_formatting.png](imgs%2Fmulti_variable_formatting.png)

** note input_variables will accept any type and attempt to format. I wanted to be able to pass strings, ints, floats etc but given its unrestricted this may break if given types such a LATENT

** primate node doesnt work however

## Known issues
Show Path seems to always hold first value it receives and then will redner a second text window with any updated values. Not sure why this is happening. Appears to be something related to PATH being a custom type