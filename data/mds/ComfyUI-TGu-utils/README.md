# TGu Utilities

**Author:** TGu-97

## Introduction

This is a set of custom nodes for ComfyUI. Mainly focus on control switches.

## Nodes

### LoRA Switch

Controls whether a LoRA Loader and its corresponding trigger prompts will be included in the queue.

Inputs:

- **model1**: The original model.
- **condition1**: The original condition.
- **model2**: The original model + LoRA.
- **condition2**: The original condition + LoRA prompts, use Conditioning (Combine) or Conditioning (Concat).

Outputs:

- **MODEL**: If the switch is OFF, output model1. Otherwise model2.
- **CONDITIONING**: If the switch is OFF, output conditioning1. Otherwise conditioning2.