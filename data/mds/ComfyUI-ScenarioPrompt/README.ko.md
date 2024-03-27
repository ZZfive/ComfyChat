# ComfyUI-ScenarioPrompt

![alt text](data/static/Scenario_prompt.png)

ScenarioPrompt는 프롬프트를 작성할 때 각 속성별로 어떤 프롬프트를 작성했는지 알기쉽게 파악할 수 있는 커스텀노드입니다.

현재 10가지 속성으로 제공되며, **Base 항목은 필수로 입력하셔야합니다.**

![alt text](data/static/Scenario_prompt_example1.png)

[CLIP Text Encode (Prompt)](https://blenderneko.github.io/ComfyUI-docs/Core%20Nodes/Conditioning/CLIPTextEncode/) 와 연결하여 사용할 수 있으며, [ComfyUI-Custom-Scripts](https://github.com/pythongosssss/ComfyUI-Custom-Scripts) 의 Show Text 항목으로 입력된 프롬프트들을 일렬로 나열해 확인해 볼 수 있습니다.

⚠️ ScenarioPrompt를 prompt에 연결하려면 반드시 CLIP Text Encode (Prompt)에 **Convert text to input** 항목에 체크하셔야합니다.

![![alt text](Scenario_prompt_caution.png)](data/static/Scenario_prompt_caution.png)

## Installation

```bash
cd path/to/comfyui/custom_nodes/
```

```bash
git clone https://github.com/Ludobico/ComfyUI-ScenarioPrompt.git
```

## Autocomplete

[ComfyUI - CustomNode Unlimited Prompt](https://github.com/PluMaZero/ComfyUI-SpaceFlower) 와 [NovelAI.app](https://github.com/gangjun06/NovelAI.app) 위 두 개의 소스코드에서 영감을 받아 만들었습니다.

자동완성에 사용되는 태그 역시 NovelAI.app 의 구글시트를 기반으로 만들어졌습니다.

![![alt text](ScenarioPrompt_autocomplete.gif)](data/static/ScenarioPrompt_autocomplete.gif)

## Reference

https://github.com/gangjun06/NovelAI.app

https://github.com/PluMaZero/ComfyUI-SpaceFlower
