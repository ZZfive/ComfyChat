# ComfyUI_kkTranslator_nodes

这些 `nodes` 主要用于把提示词从其他语言翻译成英文.
`PromptTranslateToText` 基于 `Helsinki-NLP` 的翻译模型实现的提示词翻译，不需要连网翻译

## 安装
1. 克隆这个仓库到` ComfyUI` 的 `custom_nodes` 文件夹中。
```
cd custom_nodes
git clone https://github.com/AIGCTeam/ComfyUI_kkTranslator_nodes
```
2. 运行以下命令安装依赖
```
python -m pip install -r requirements.txt
```
3. 重启你的 ComfyUI

## 使用
下载这个工作流查看演示: [marian_mt_workflow](./marian_mt_workflow.json)

![Alt text](image.png)

### 百度翻译插件
如果你有注册百度翻译开发者账户，可以使用百度翻译插件 `PromptBaiduFanyiToText`。百度翻译支持的翻译内容比本地模型更好，也更快，但是需要在 `PromptBaiduFanyiToText` 中配置你的百度翻译账户信息。注册方法: [https://fanyi-api.baidu.com/doc/12](https://fanyi-api.baidu.com/doc/12)

![promptbaidu](image-1.png)

## 重要!!

国内无法由于存在 huggingface 的服务器问题,麻烦请参考: https://hf-mirror.com/ 启动 comfyui

```
export HF_ENDPOINT=https://hf-mirror.com
python main.py

# or 
 HF_ENDPOINT=https://hf-mirror.com python main.py
```