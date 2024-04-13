![afbeelding](https://github.com/TheBarret/ZSuite/assets/25234371/309a5f2c-81cb-496c-bb79-a34b56d08807)


Version: 2.0.0
- 10-11-2023 - Initial Release
- 11-11-2023 - Added Seed Modifier
- 12-11-2023 - Modified Seed node to accept expressions
- 13-11-2023 - Added Latent Noise Provider (Transforming Ambient RF Signals using `RTL_TCP` as noise)


# ZSuite - Prompter

This node uses files to randomize pre-defined sorted subjects of random things.

Example line:

`__preamble__ painting by __artist__`

This prompt will be processed with random line form the file called `preamble.txt` and `artists.txt` in the folder:

`.\comfyui\custom_nodes\Zephys\nodes\blocks\*.txt`

You can create new or change any `txt` files in this folder to customize your wishes

The node uses a `trigger` as input from any `integer` value type, to enforce a new prompt output.
I often myself use a `Counter` node that comes almost by default for math operations, it allows
me to make sure the `Prompter` gets processed each time you hit `Generate`.


# ZSuite - RF Node (Testing Phase)

- *This section explains the functioning of the RF Node within ZSuite, emphasizing the need for an RTL-SDR device, server setup, and predefined configurations.
the rtl_tcp is an I/Q spectrum server for RTL2832 based DVB-T receivers*
- *This node is still a prototype, features may change in the future*

The protocol netcode is adopted from and written by `Paul Tagliamonte` (2020-11-03: https://hz.tools/rtl_tcp/)


![afbeelding](https://hackaday.com/wp-content/uploads/2017/09/dongle.png)


**Utilizing RF Ambient Noise for Robust Randomization:**

Incorporating RF ambient noise provides a robust source of random data, enhancing the capabilities of our randomizer provider. By tuning our device to any available frequency, we harness the inherent unpredictability of ambient RF signals. This approach ensures a diverse and reliable pool of random data. Users have the flexibility to fine-tune their devices to specific frequencies, even capturing signals from existing stations and utilizing their amplitudes or signal characteristics for the randomization process. This strategic utilization of RF ambient noise elevates the effectiveness and versatility of our randomization provider within the system.

![afbeelding](https://github.com/TheBarret/ZSuite/assets/25234371/44123c41-493e-41bf-815b-7c700da6d134)



**Server Hosting and RTL_TCP Service:**

To facilitate communication with the RTL-SDR device, a server must be set up. The RTL_TCP service is employed for this purpose. Here's a breakdown of the setup:

1. **RTL_TCP Installation:**
   - Follow the Debian-specific instructions provided in the [RTL_TCP Manpage](https://manpages.debian.org/testing/rtl-sdr/rtl_tcp.1.en.html).
   - Execute the command: `rtl_tcp -a <ip> -p <port> -d 0` to initiate the service on the specified IP and port.
   - The IP is to be expected either localhost (safe) or outside ip address for servibbg outside your network.
   - Port can be anything you wish, I do advice to use a high-range, because in Linux environment most ports lower then `< 1024` are reserved and restricted.

![afbeelding](https://github.com/TheBarret/ZSuite/assets/25234371/fd5e517c-c3bd-4ad6-a219-c61648bf757c)

![afbeelding](https://github.com/TheBarret/ZSuite/assets/25234371/c333f042-ff4c-41f7-9581-c667fe02db82)

**Why Server Setup is Necessary:**

The RTL-SDR device communicates via TCP/IP, necessitating the establishment of a server. This server acts as an intermediary, allowing ZSuite to access the data stream from the RTL-SDR device. The specified IP and port parameters ensure a seamless connection between the RF Node in ZSuite and the RTL-SDR device.

**Predefined Configurations and Latent Noise Processing:**

Within the ZSuite framework, the RF Node operates on pre-defined configurations and processes data using the following steps:

1. **Data Capture Parameters:**
   - The RF Node captures data at a rate of (default) `4096 bytes per cycle`.
   - The duration parameter governs capture length of the capturing.

![afbeelding](https://github.com/TheBarret/ZSuite/assets/25234371/b13f7ca0-5b76-4210-9c2d-0636c4400721)


2. **Latent Noise Processing:**
   - If the captured data length is smaller than the specified Latent shape, it resets to index 0.
   - The captured data undergoes processing using the `numpy` library to perform normalization.
   - Normalization involves scaling and injecting noise into latent samples, ensuring the data is in a workable format.

**Configurability:**

Users can customize default device parameters by editing the `ZS_Rtlsdr.py`,

this file is located at `.\comfyui\custom_nodes\Zephys\nodes\`.

The constant definition header in this file contains configurable settings for the RF Node.

**SDR RTL Device Parameters Explained**

To better understand the RF-based settings in ZSuite, consider the following explanations:

1. **Frequency (Frequnecy):**
   - *Definition:* Represents the operating frequency of the RTL-SDR device.
   - *Range:* Typically between 5KHz and 1.7GHz.
   - *Unit:* Measured in Hertz (`Hz`).

2. **Gain Values:**
   - *Definition:* Refers to the amplification applied to the received signal.
   - *Options:* 0.0, 0.9, 1.4, 2.7, 3.7, 7.7, 8.7, 12.5, 14.4, 15.7, 16.6, 19.7, 20.7, 22.9, 25.4, 28.0, 29.7, 32.8, 33.8, 36.4, 37.2, 38.6, 40.2, 42.1, 43.4, 43.9, 44.5, 48.0, 49.6.
   - *Default:* 0 (auto).

3. **Samplerates:**
   - *Definition:* Indicates the rate at which the RTL-SDR device samples the incoming signal.
   - *Options:* 0.25, 1.024, 1.536, 1.792, 1.92, 2.048, 2.16, 2.56, 2.88, 3.2 MSps (Mega Samples per second).
   - *Default:* 1.024 MSps.

**Additional Information:**

**Frequency Range Explanation:**
  - The frequency range is the span of frequencies the RTL-SDR device can capture. It is crucial to set this parameter based on the type of signals you intend to receive.
    
    (Example: one could tune in on `118Mhz` or `118000000`hz and in this range most if not all, local ATC flight communications take place)

**Gain Values Significance:**
  - Gain values control the amplification level. Higher gain may improve weak signal reception but can introduce noise. Auto (default) adjusts gain automatically.

The series can be expressed as a piecewise function that combines two arithmetic progressions.

```
def series_function(x):
    if x % 2 == 0:
        return 0.9 * x
    else:
        return 3.7 + 4.0 * ((x - 1) / 2)
```
![afbeelding](https://github.com/TheBarret/ZSuite/assets/25234371/823529b3-aabe-4f57-a70a-fc1c1d8f95ff)

**Samplerates Impact:**
  - Samplerates determine how many samples per second the RTL-SDR device collects. Higher rates provide more detailed signal information but may increase processing load.

Understanding and adjusting these settings allow users to optimize the RTL-SDR device for specific signal types and environmental conditions, enhancing the effectiveness of RF signal processing in ZSuite.

By following these steps and understanding the underlying processes, users can effectively set up and utilize the RF Node in ZSuite for their specific needs.
