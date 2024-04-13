# comfyui_otonx_nodes

Repository: Otonx's Custom Nodes for ComfyUI.

**Note:** Nodes in this repository are subject to intermittent changes. Please review and test nodes thoroughly prior to integration.

**Installation**

To install the custom nodes, you can follow the following steps:
- cd custom_nodes
- git clone https://github.com/budihartono/comfyui_otonx_nodes.git
- Restart ComfyUI

---

### OTX KSampler Feeder

**Description:** Centralized value storage for input parameters intended for the KSampler (Advanced) nodes. Designed based on the workflow outlined in [(Part 4) Advanced SDXL Workflows in ComfyUI](https://followfoxai.substack.com/i/136667610/changes-to-the-previous-workflow).

![KSampler (Advanced) Inputs](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F07e38aa6-a492-474d-8c86-78b050bdec2b_1428x391.png)

![KSampler (Advanced) Inputs](https://substackcdn.com/image/fetch/w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2Fa86bfe0f-60cf-4c64-bbc9-c2ed8b3b4beb_1417x355.png)

**Outputs:**
- **seed:** Allows identical `noise_seed` values for both Base KSampler and Refiner KSampler nodes.
- **steps:** Maintains a consistent `steps` value between the base and refiner samplers.
- **cfg:** Allows identical `cfg` values to be set for both base and refiner samplers.
- **base_steps_portion:** Outputs a value that can synchronize the `start_at_step` of the refiner with the `end_at_step` of the base.

**Utility:** Reduces the need for multiple primitive nodes containing singular values, centralizing configuration.

---

### OTX Integer Multiple Inputs

**Description:** This node series is specialized for receiving multiple integer inputs, optimizing data input processes while ensuring consistent data type integration.

**Functionality:** The node provides several slots, each designed to accept integer values. Users have the flexibility to manually set each slot by directly typing in the integer values, streamlining the input process or connecting the inputs or outputs to other nodes,. Each slot acts as a dedicated integer input that can be seamlessly connected to downstream nodes.

**Utility:** It streamlines node setups by consolidating multiple integer input slots into a single node, reducing the need for individual integer nodes.

- **OTX Integer Multiple Inputs 4:** Features a 4-slot configuration for integer inputs.
- **OTX Integer Multiple Inputs 5:** Features a 5-slot configuration for integer inputs.
- **OTX Integer Multiple Inputs 6:** Features a 6-slot configuration for integer inputs.

---

### OTX Versatile Multiple Inputs

**Description:** A dynamic node series designed for diverse data input configurations. It can accommodate varying data types, granting users the flexibility to define and manage multiple data inputs within a singular node.

**Functionality:** Each slot within the node, implemented as a multiline textbox to handle longer strings, can be configured to accept one of three data types: INT, FLOAT, or STRING. In addition to connecting the inputs or outputs to other nodes, users can manually set data by typing into the multiline textboxes, ensuring both the value and its type are correctly configured for downstream processing.

**Utility:** This node series offers an integrated platform for diverse data inputs, negating the need for multiple nodes dedicated to distinct data types.

- **OTX Versatile Multiple Inputs 4:** A 4-slot design, where each multiline textbox slot is configurable to INT, FLOAT, or STRING.
- **OTX Versatile Multiple Inputs 5:** A 5-slot design, where each multiline textbox slot is configurable to INT, FLOAT, or STRING.
- **OTX Versatile Multiple Inputs 6:** A 6-slot design, where each multiline textbox slot is configurable to INT, FLOAT, or STRING.

<img src="https://github.com/budihartono/comfyui_otonx_nodes/blob/main/docs/img/otx_versatile_inputs.png" width="40%" height="40%">
