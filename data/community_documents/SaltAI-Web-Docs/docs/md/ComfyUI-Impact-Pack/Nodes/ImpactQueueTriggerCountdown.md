# Queue Trigger (Countdown)
## Documentation
- Class name: `ImpactQueueTriggerCountdown`
- Category: `ImpactPack/Logic/_for_test`
- Output node: `True`

This node is designed to manage a countdown mechanism within a queue system, allowing for the triggering of events based on a count relative to a total, with the option to reset or continue the count based on a mode setting.
## Input types
### Required
- **`signal`**
    - Acts as a pass-through signal that can be modified or utilized in the countdown process.
    - Comfy dtype: `*`
    - Python dtype: `object`
- **`count`**
    - Represents the current count in the countdown process, influencing whether the countdown continues or resets.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`total`**
    - Defines the total count required to trigger a reset in the countdown process.
    - Comfy dtype: `INT`
    - Python dtype: `int`
- **`mode`**
    - Determines the behavior of the countdown, either continuing the count or triggering a reset based on the count's relation to the total.
    - Comfy dtype: `BOOLEAN`
    - Python dtype: `bool`
## Output types
- **`signal_opt`**
    - Comfy dtype: `*`
    - The modified or unmodified pass-through signal after the countdown process.
    - Python dtype: `object`
- **`count`**
    - Comfy dtype: `INT`
    - The updated count after the countdown operation.
    - Python dtype: `int`
- **`total`**
    - Comfy dtype: `INT`
    - The total count required for a reset, unchanged by the operation.
    - Python dtype: `int`
## Usage tips
- Infra type: `CPU`
- Common nodes: unknown


## Source code
```python
class ImpactQueueTriggerCountdown:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "signal": (any_typ,),
                    "count": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "total": ("INT", {"default": 10, "min": 1, "max": 0xffffffffffffffff}),
                    "mode": ("BOOLEAN", {"default": True, "label_on": "Trigger", "label_off": "Don't trigger"}),
                    },
                "hidden": {"unique_id": "UNIQUE_ID"}
                }

    FUNCTION = "doit"

    CATEGORY = "ImpactPack/Logic/_for_test"
    RETURN_TYPES = (any_typ, "INT", "INT")
    RETURN_NAMES = ("signal_opt", "count", "total")
    OUTPUT_NODE = True

    def doit(self, signal, count, total, mode, unique_id):
        if count < total - 1 and (mode):
            PromptServer.instance.send_sync("impact-node-feedback",
                                            {"node_id": unique_id, "widget_name": "count", "type": "int", "value": count+1})
            PromptServer.instance.send_sync("impact-add-queue", {})
        if count >= total - 1:
            PromptServer.instance.send_sync("impact-node-feedback",
                                            {"node_id": unique_id, "widget_name": "count", "type": "int", "value": 0})

        return (signal, count, total)

```
