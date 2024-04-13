# ComfyUI Portrait Master 简体中文版


![Dingtalk_20231221171244](https://github.com/ZHO-ZHO-ZHO/comfyui-portrait-master-zh-cn/assets/140084057/f14a31f6-56f0-4e3e-9bf0-5a7a209175bd)


## 项目介绍 | Info

- 人物肖像提示词生成模块，优化肖像生成，选择永远比填空更适合人类！

- 优化 + 汉化 自 [ComfyUI Portrait Master](https://github.com/florestefano1975/comfyui-portrait-master.git)

- 版本：V2.2

- 版本说明：版本越高内容越多，但随着提示词逐渐增多，每项参数的最终效果可能会被削弱，因此并非版本越高越好用，请选择适合自己的版本

## 参数说明 | Parameters

- 镜头类型：头像、肩部以上肖像、半身像、全身像、脸部肖像
- 性别：女性、男性
- 国籍_1：193个国家可选
- 国籍_2：193个国家可选
- 体型:瘦、正常、超重等4种 🆕🆕
- 姿势：回眸、S曲线、高级时尚等18种 🆕🆕
- 眼睛颜色：琥珀色、蓝色等8种 🆕
- 面部表情：开心、伤心、生气、惊讶、害怕等24种
- 脸型：椭圆形、圆形、梨形等12种
- 发型：法式波波头、卷发波波头、不对称剪裁等20种
- 头发颜色：金色、栗色、灰白混合色等9种 🆕
- 胡子：山羊胡、扎帕胡等20种 🆕🆕
- 灯光类型：柔和环境光、日落余晖、摄影棚灯光等32种 🆕
- 灯光方向：上方、左侧、右下方等10种 🆕
- 起始提示词：写在开头的提示词
- 补充提示词：写在中间用于补充信息的提示词
- 结束提示词：写在末尾的提示词
- 提高照片真实感：可强化真实感 🆕
- 负面提示词：新增负面提示词输出 🆕

## 提示词合成顺序 | Prompt composition order
- 起始提示词
- 镜头类型 + 镜头权重
- 国籍 + 性别 + 年龄
- 体型 🆕🆕
- 姿势 🆕🆕
- 眼睛颜色 🆕
- 面部表情 + 面部表情权重
- 脸型
- 发型
- 头发颜色 🆕
- 胡子 🆕🆕
- 头发蓬松度
- 补充提示词
- 皮肤细节
- 皮肤毛孔
- 皮肤瑕疵
- 痘痘 🆕🆕
- 皱纹 🆕🆕
- 小麦色肤色 🆕🆕
- 酒窝
- 雀斑
- 痣
- 眼睛细节
- 虹膜细节
- 圆形虹膜
- 圆形瞳孔
- 面部对称性
- 灯光类型 + 灯光方向 🆕
- 结束提示词
- 提高照片真实感 🆕

## 姿势库 | Model Pose Library

特别提醒：由于肖像大师的本质是提示词，因此想要通过纯提示词实现姿势的稳定生成需要大量抽卡才能实现，这是我测试抽卡了一下午精选的结果，所以建议配合 openpose 实现姿势的精确控制，别为难自己！

![poselist_](https://github.com/ZHO-ZHO-ZHO/comfyui-portrait-master-zh-cn/assets/140084057/0eac37da-6aee-4591-9755-19e3b317724c)


## 自定义 | Customizations

可将需要自定义增加的内容写到lists文件夹中对应的json文件里（如发型、表情等）

## 使用建议 | Practical advice

- 注意：随着提示词逐渐增多，每项参数的最终效果可能会被削弱，不建议满铺所有参数
  
- 皮肤和眼睛细节等参数过高时可能会覆盖所选镜头的设置。在这种情况下，建议减小皮肤和眼睛的参数值，或者插入否定提示(closeup, close up, close-up:1.5)，并根据需要修改权重。

- 要实现完美的姿势控制，请配合 ControlNet 使用，同时将镜头类型设置为空（-）

## 安装 | Install

- 推荐使用 ComfyUI Manager 安装

- 手动安装：
    1. `cd custom_nodes`
    2. `git clone https://github.com/ZHO-ZHO-ZHO/comfyui-portrait-master-zh-cn.git`
    3. 重启 ComfyUI

## 工作流 | Workflow

### V2.2工作流

- [V2.2 For SD1.5 or SDXL](https://github.com/ZHO-ZHO-ZHO/comfyui-portrait-master-zh-cn/blob/main/workflows/Portrait%20Master%20%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87%E7%89%88%20V2.2%E3%80%90Zho%E3%80%91.json)

![Dingtalk_20231221171315](https://github.com/ZHO-ZHO-ZHO/comfyui-portrait-master-zh-cn/assets/140084057/f36c43f7-5381-470b-a5f5-8abed834e2e2)

### V2.0工作流

- [V2.0 For SD1.5 or SDXL](https://github.com/ZHO-ZHO-ZHO/comfyui-portrait-master-zh-cn/blob/main/workflows/Portrait%20Master%20%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87%E7%89%88%20V2.0%E3%80%90Zho%E3%80%91.json)

![Dingtalk_20231218163927](https://github.com/ZHO-ZHO-ZHO/comfyui-portrait-master-zh-cn/assets/140084057/606e1ef4-429c-4f8d-99fb-0a19f2350d0e)

- [V2.0 For SDXL Turbo（non-commercial）](https://github.com/ZHO-ZHO-ZHO/comfyui-portrait-master-zh-cn/blob/main/workflows/Portrait%20Master%20%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87%E7%89%88%20SDXL%20Turbo%20V2.0%E3%80%90Zho%E3%80%91.json)

![Dingtalk_20231218165449](https://github.com/ZHO-ZHO-ZHO/comfyui-portrait-master-zh-cn/assets/140084057/e0b188af-7d0a-47b8-8327-13dd630cea91)

- [V2.0 for SAG + SVD 视频工作流](https://github.com/ZHO-ZHO-ZHO/comfyui-portrait-master-zh-cn/blob/main/workflows/Portrait%20Master%20%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87%E7%89%88%20V2.0%20%2B%20SAG%20%2B%20SVD%E3%80%90Zho%E3%80%91.json)


https://github.com/ZHO-ZHO-ZHO/comfyui-portrait-master-zh-cn/assets/140084057/8e3915be-2d45-4f94-af0c-0a270378712b

![Dingtalk_20231218185612](https://github.com/ZHO-ZHO-ZHO/comfyui-portrait-master-zh-cn/assets/140084057/e9316a7a-dbe5-4e20-bd50-1e622551c7ab)


### V1.0工作流

- [SD1.5 or SDXL](https://github.com/ZHO-ZHO-ZHO/comfyui-portrait-master-zh-cn/blob/main/workflows/Portrait%20Master%20%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87%E7%89%88%E3%80%90Zho%E3%80%91.json)

![image](https://github.com/ZHO-ZHO-ZHO/comfyui-portrait-master-zh-cn/assets/140084057/e1269817-36e6-4f20-92f6-7119128b65d4)


- [SDXL Turbo（non-commercial）](https://github.com/ZHO-ZHO-ZHO/comfyui-portrait-master-zh-cn/blob/main/workflows/Portrait%20Master%20%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87%E7%89%88%20SDXL%20Turbo%E3%80%90Zho%E3%80%91.json)

![image](https://github.com/ZHO-ZHO-ZHO/comfyui-portrait-master-zh-cn/assets/140084057/459162f0-a079-42af-990b-e916f32a0ff7)


## 更新日志 | Changelog

20231221

- 更新为V2.2版，新增6项参数：
    - 体型（4种）
    - 姿势（18种）
    - 胡子（20种）
    - 痘痘
    - 皱纹
    - 小麦色肤色
 
- 已登陆 manager 不用手动安装了

20231218

- 更新为V2.0版，新增6项参数，扩充2项参数，优化代码：
    - 眼睛颜色（8种）
    - 头发颜色（9种）
    - 灯光类型（32种）
    - 灯光方向（10种）
    - 提高照片真实感
    - 负面提示词
    - 镜头类型（+3种）
    - 发型（+19种）

![Dingtalk_20231218164020](https://github.com/ZHO-ZHO-ZHO/comfyui-portrait-master-zh-cn/assets/140084057/38d305cb-64f3-4dcf-a389-5ad3f84be7b3)

20231216

- 完成代码优化，将原本读取txt文件调整成读取json文件，更加方便使用、自定义和扩展
  ![image](https://github.com/ZHO-ZHO-ZHO/comfyui-portrait-master-zh-cn/assets/140084057/7b183c08-a95f-4464-9e51-979894cb2b60)

20231215

- 对 [ComfyUI Portrait Master](https://github.com/florestefano1975/comfyui-portrait-master.git) 完成汉化


## Stars 

[![Star History Chart](https://api.star-history.com/svg?repos=ZHO-ZHO-ZHO/comfyui-portrait-master-zh-cn&type=Date)](https://star-history.com/#ZHO-ZHO-ZHO/comfyui-portrait-master-zh-cn&Date)

<!---
[![Star History Chart](https://api.star-history.com/svg?repos=ZHO-ZHO-ZHO/comfyui-portrait-master-zh-cn&type=Timeline)](https://star-history.com/#ZHO-ZHO-ZHO/comfyui-portrait-master-zh-cn&Timeline)
--->

## Credits

[ComfyUI Portrait Master](https://github.com/florestefano1975/comfyui-portrait-master.git)


