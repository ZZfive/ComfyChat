# ComfyUI_ChatGLM_API

**注意:**  
这是可以在ComfyUI里调用Chatglm4、3等模型的API插件，用来翻译、描述图片等等，类似OpenAI API 或Claude API。如果你只想调用Api查询资料或者娱乐,可选Original_language,作为你的母语输出. 限于中国的法律法规,所有NSFW内容可能会被过滤掉,如果违规上传NSFW的内容,你的zhipuai账户可能因此被封禁. 

**Note:**   
This is an API plugin that can be used in ComfyUI to call models such as Chatglm4 and 3 for translating, describing images, and more, similar to OpenAI API or Claude API. If you only want to call Api to search for information or entertainment, you can choose Original_language as your native language output Due to Chinese laws and regulations, all NSFW content may be filtered out. If you upload NSFW content in violation of regulations, your Zhipuai account may be banned as a result

**A  使用方法：**  

1、在..\ComfyUI\custom_nodes目录下，打开git，使用git clone https://github.com/smthemex/ComfyUI_ChatGLM_API.git 安装插件；  
2、打开ComfyUI\custom_nodes\ComfyUI-CHATGLM4-API下的config文件，把api_key里替换成你申请到的api_key，并保存；  
3、在comfyUI中找到chatglm_API节点再使用；  
4、网络鉴权或许会有延时……  


**B 注意事项:**  

这是一个调用ChATGLM-4，GLM-3-Turbo,CHATGLM-4V的ComfyUI节点，在使用此节点之前，你需要去智谱AI的官网 https://open.bigmodel.cn  ，注册并申请API_key,新用户送200万tokens,实名认证再送300万tokens，有效期1个月。  

如果是付费使用，费用如下:  
GLM-4：  0.1元 / 千tokens，    
GLM-4V： 0.1元 / 千tokens，    
GLM-3-Turbo：  0.005元 / 千tokens， 推荐用GLM-3-Turbo，主打便宜    
------

**C 更新内容：**   

1、加入language可选，language控制输出的语言类别，origin为初始输入语言  

2、加入了max_tokens和temperature参数，max_tokens可以控制输出文本的数量，temperature参数越大，随机性越强，越小，准确度越高  
Added max_tokens and temperature parameters, which can control the quantity of output text. The larger the temperature parameter, the stronger the randomness, and the smaller the accuracy  

3、加入翻译为英文选项，开启时仅翻译为提示词英文，此时对话模式下 English和origin_language不起作用。  

**D 示例：**    

1、txt2txt节点一: 文本翻译,故事描述等...   

可选ChATGLM-4，GLM-3-Turbo模型，prompt你可以直接输入中文或任何其他文字，让然GLM4帮你写提示词，中国大陆用户方便点，外网用好像也可以，比较麻烦而已。    
max_tokens控制输出文本的数量； 
temperature控制随机性；  
language控制对话输出的语言类别，选择origin时，对话输出结果为初始输入语言，无法作为prompt使用。   

<span style="color:#333333"><img src="https://github.com/smthemex/ComfyUI_ChatGLM_API/blob/main/workflow/txt2txt2.png" width="50%"></span>

2、img2txt节点二: 图片描述,图片分析等    

用的是CHATGLM-4V模型，类似WD14一类的图片反推，只是用CHATGLM-4V帮你反推,    

注意：图片载入前,最好使用"图片剪裁或缩放节点",减少图片编码为Base64数据后的长度,官方只接受小于5M的图片.      
  
<span style="color:#333333"><img src="https://github.com/smthemex/ComfyUI_ChatGLM_API/blob/main/workflow/img2txt2.png" width="50%"></span>

3、翻译节点开启，同一提示词不同输出示例    

以下为仅翻译模式，输出prompt翻译结果：   

<span style="color:#333333"><img src="https://github.com/smthemex/ComfyUI_ChatGLM_API/blob/main/workflow/trans_only_examples.png" width="50%"></span>

以下为关闭翻译，chat模式，输出英文对话：    

<span style="color:#333333"><img src="https://github.com/smthemex/ComfyUI_ChatGLM_API/blob/main/workflow/reply%20in%20english_examples.png" width="50%"></span>

**E 其他:**    
如果有空，会继续补全其他的模型。  

智谱其实做了ChatGLM_API接口的SDK https://github.com/zhipuai/zhipuai-sdk-python  ，有兴趣可以去尝试。    
迁移到ComfyUI时，我发现SDK在ComfyUI的节点里好像不大管用，所以这个节点未使用智谱的SDK，而是用常规的HTTP调用API接口。    


----------
致谢: 本节点参考了-ZHO-的基于"通义千问API"开发的ComfyUI节点部分代码 https://github.com/ZHO-ZHO-ZHO/ComfyUI-Qwen-VL-API  
-----
