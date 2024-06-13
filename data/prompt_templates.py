

# 构建v1数据集时，直接将从自定义节点项目拉取的md文件中的内容喂给LLMs，使其自己生成问答数据，以下是当时测试的模板
# template = '''
# I need to build a llm fine-tuning dataset. You need to understand the content of the document I input, then construct several pairs of question and answer data yourself, and return them in json format.\n---\nOnly question and answer data in json format is returned. The file currently being passed in is about {0}, the specific contents are as follows: {1}
# '''

template =  '''
I need to build a llm fine-tuned dataset. You need to understand the document content I input, then construct the question and answer data pair yourself, and return it in json format. The documentation I'm passing on is all about ComfyUI (a GUI that uses a stable diffusion model to generate images and videos) and custom nodes or plugins that extend its functionality. When building question and answer data, it must be clear whether the subject is for ComfyUI or a specific custom node or plug-in. The subject in the Q&A data must carry the specific name of the node or plug-in, such as the \"ComfyUI-Manager\" extension; do not just use \"extension\" or \"custom node\" as the subject, which does not indicate that the question is about the specific name of the Node or plug-in. Note that I will tell you the described subject name before passing in the specific document content, and you can use it directly when building question and answer data. Ensure that the constructed question and answer data cover all the content of the text as much as possible. Please ensure that the output json data format is correct. Do not miss necessary symbols, but do not add unnecessary symbols. The file currently being passed in is about {0}, the specific contents are as follows: {1}
'''

# template = '''
# I need to build a llm fine-tuned dataset. You need to understand the document content I input, then construct the question and answer data pair yourself, and return it in json format. The documentation I'm passing on is all about ComfyUI (a GUI that uses a stable diffusion model to generate images and videos) and custom nodes or plugins that extend its functionality. When building question and answer data, it must be clear whether the subject is for ComfyUI or a specific custom node or plug-in. The subject in the Q&A data must carry the specific name of the node or plug-in, such as the \"ComfyUI-Manager\" extension; do not just use \"extension\" or \"custom node\" as the subject, which does not indicate that the question is about the specific name of the Node or plug-in. Note that I will tell you the described subject name before passing in the specific document content, and you can use it directly when building question and answer data. The constructed question and answer data should include English question and answer data and Chinese question and answer data respectively, and cover all the content of the text as much as possible. Please ensure that the output json data format is correct. Do not miss necessary symbols, but do not add unnecessary symbols. \n---\nOnly question and answer data in json format is returned. The returned content is as follows:\n[\n          {{\n              \"english\": [\n                  {{\n                      \"question\": \"...\",\n                      \"answer\": \"...\"\n                  }},\n                  {{\n                      \"question\": \"...\",\n                      \"answer\": \"...\"\n                  }}\n              ]\n          }},\n          {{\n              \"chinese\": [\n                  {{\n                      \"question\": \"...\",\n                      \"answer\": \"...\"\n                  }},\n                  {{\n                      \"question\": \"...\",\n                      \"answer\": \"...\"\n                  }}\n              ]\n          }}\n]\n---\nThe file currently being passed in is about {0}, the specific contents are as follows: {1}
# '''

# prefix = """
# I need to build a llm fine-tuned dataset. You need to understand the document content I input, then construct the question and answer data pair yourself, and return it in json format. The documentation I'm passing on is all about ComfyUI (a GUI that uses a stable diffusion model to generate images and videos) and custom nodes or plugins that extend its functionality. When building question and answer data, it must be clear whether the subject is for ComfyUI or a specific custom node or plug-in. The subject in the Q&A data must carry the specific name of the node or plug-in, such as the \"ComfyUI-Manager\" extension; do not just use \"extension\" or \"custom node\" as the subject, which does not indicate that the question is about the specific name of the Node or plug-in. Note that I will tell you the described subject name before passing in the specific document content, and you can use it directly when building question and answer data. The constructed question and answer data should include English question and answer data and Chinese question and answer data respectively, and cover all the content of the text as much as possible.\n---\nOnly question and answer data in json format is returned. The returned content is as follows:\n[\n          {\n              \"english\": [\n                  {\n                      \"question\": \"...\",\n                      \"answer\": \"...\"\n                  },\n                  {\n                      \"question\": \"...\",\n                      \"answer\": \"...\"\n                  }\n              ]\n          },\n          {\n              \"chinese\": [\n                  {\n                      \"question\": \"...\",\n                      \"answer\": \"...\"\n                  },\n                  {\n                      \"question\": \"...\",\n                      \"answer\": \"...\"\n                  }\n              ]\n          }\n]\n---\n
# """

# 构建v2数据集使用得提示词模板
# 从ComfyUI-docs[https://github.com/BlenderNeko/ComfyUI-docs]使用LLMs构建问答数据的提示词模板
system_prompt1 = "I want you to play the role of a question-answer data builder and generate reasonable question-answer data pairs based on the text I passed in. Don't make up information that is not in the passed in text. You need adjust the number of generated question-answer data pairs based on the length of the passed in text, but generate at least seven question-answer data pairs each time."

template1 = '''
# CONTEXT #
I want to fine-tune a large language model. I need to build a fine-tuning dataset, which requires generating a lot of question-answer data pairs.
#############

# OBJECTIVE #
You need to understand the document content I input, then construct the question and answer data pair yourself, and return it in json format. The documentation I'm passing on is all about ComfyUI (a GUI that uses a stable diffusion model to generate images and videos) and custom nodes or plugins that extend its functionality. When building question and answer data, it must be clear whether the subject is for ComfyUI or a specific custom node or plug-in. The subject in the Q&A data must carry the specific name of the node or plug-in, such as the \"ComfyUI-Manager\" extension; do not just use \"extension\" or \"custom node\" as the subject, which does not indicate that the question is about the specific name of the Node or plug-in. You need adjust the number of generated question-answer data pairs based on the length of the passed in text, but generate at least seven question-answer data pairs each time. Note that I will tell you the described subject name before passing in the specific document content, and you can use it directly when building question and answer data. Ensure that the constructed question and answer data cover all the content of the text as much as possible. Please ensure that the output json data format is correct. Do not miss necessary symbols, but do not add unnecessary symbols.
#############

# TONE #
Professional, technical
#############

# RESPONSE #
[
    {{
        "question": "What is cg-noise?",
        "answer": "cg-noise is a custom node in ComfyUI that replaces KSampler and KSampler Advanced, allowing for small variations in the initial noise."
    }},
    ...,
    {{
        "question": "How does cg-noise generate variations in images?",
        "answer": "cg-noise generates variations in images by using a weight `x` and two seeds. It generates the noise based on `random_based_on(variation_seed) * x + random_based_on(seed) * (1-x)`."
    }}
]
#############

The file currently being passed in is about {0}, the specific contents are as follows: {1}
'''

# 从SaltAI-Web-Docs[https://github.com/get-salt-AI/SaltAI-Web-Docs]使用LLMs构建问答数据的提示词模板
system_prompt2 = "I want you to play the role of a question-answer data builder and generate reasonable question-answer data pairs based on the text I passed in. Don't make up information that is not in the passed in text. You need adjust the number of generated question-answer data pairs based on the length of the passed in text, but generate at least seven question-answer data pairs each time."

template2 = '''
# CONTEXT #
I want to fine-tune a large language model. I need to build a fine-tuning dataset, which requires generating a lot of question-answer data pairs.
#############

# OBJECTIVE #
You need to understand the document content I input, then construct the question and answer data pair by yourself, return it in json format. Here are some things to note:
1. The documentation I'm passing on is all about ComfyUI (a GUI that uses a stable diffusion model to generate images and videos) and custom nodes or plugins that extend its functionality. When building question and answer data, it must be clear whether the subject is for ComfyUI or a specific custom node or plug-in. The subject in the Q&A data must carry the specific name of the node or plug-in, such as the \"ComfyUI-Manager\" extension; do not just use \"extension\" or \"custom node\" as the subject, which does not indicate that the question is about the specific name of the Node or plug-in.
2. You need adjust the number of generated question-answer data pairs based on the length of the passed in text, but generate at least seven question-answer data pairs each time.
3. Note that I will tell you the described subject name before passing in the specific document content, and you can use it directly when building question and answer data. Ensure that the constructed question and answer data cover all the content of the text as much as possible.
4. Do not miss necessary symbols, but do not add unnecessary symbols.
5. When generating question-answer pairs, do not just start from a specific subject, but also ask questions that point to the subject in the input text from the characteristics and phenomena. For example, the LoadImage node can load images, so do not just generate questions like "What can the LoadImage node do?", but also generate questions like "What nodes can load images?"
6. Please ensure that the output json data format is correct. 
#############

# TONE #
Professional, technical
#############

# RESPONSE #
[
    {{
        "question": "What is cg-noise?",
        "answer": "cg-noise is a custom node in ComfyUI that replaces KSampler and KSampler Advanced, allowing for small variations in the initial noise."
    }},
    ...,
    {{
        "question": "How does cg-noise generate variations in images?",
        "answer": "cg-noise generates variations in images by using a weight `x` and two seeds. It generates the noise based on `random_based_on(variation_seed) * x + random_based_on(seed) * (1-x)`."
    }}
]
#############

The file currently being passed in is about {0}, the specific contents are as follows: {1}
'''

system_prompt2_index = "I want you to play the role of a question-answer data builder and generate reasonable question-answer data pairs based on the text I passed in. Don't make up information that is not in the passed in text. You need adjust the number of generated question-answer data pairs based on the length of the passed in text, but generate at least five question-answer data pairs each time."

template2_index = '''
# CONTEXT #
I want to fine-tune a large language model. I need to build a fine-tuning dataset, which requires generating a lot of question-answer data pairs.
#############

# OBJECTIVE #
You need to understand the document content I input, then construct the question and answer data pair by yourself, return it in json format. Here are some things to note:
#############

# NOTICE #
1. The Markdown documentation I'm passing on is all about ComfyUI (a GUI that uses a stable diffusion model to generate images and videos) and custom nodes or plugins that extend its functionality. When building question and answer data, it must be clear whether the subject is for ComfyUI or a specific custom node or plug-in. The subject in the Q&A data must carry the specific name of the node or plug-in, such as the \"ComfyUI-Manager\" extension; do not just use \"extension\" or \"custom node\" as the subject, which does not indicate that the question is about the specific name of the Node or plug-in.
2. You need adjust the number of generated question-answer data pairs based on the length of the passed in text, but generate at least five question-answer data pairs each time.
3. Note that I will tell you the described subject name before passing in the specific document content, and you can use it directly when building question and answer data. Ensure that the constructed question and answer data cover all the content of the text as much as possible.
4. Do not generate any question-answer pairs about "Licenses" and "Commit Version"
5. Do not generate question-answer pairs for "objects such as images or videos represented by relative positions" in documents
6. Do not miss necessary symbols, but do not add unnecessary symbols.
7. When generating question-answer pairs, do not just start from a specific subject, but also ask questions that point to the subject in the input text from the characteristics and phenomena. For example, the LoadImage node can load images, so do not just generate questions like "What can the LoadImage node do?", but also generate questions like "What nodes can load images?"
8. Please ensure that the output json data format is correct. 
#############

# TONE #
Professional, technical
#############

# RESPONSE #
[
    {{
        "question": "What is cg-noise?",
        "answer": "cg-noise is a custom node in ComfyUI that replaces KSampler and KSampler Advanced, allowing for small variations in the initial noise."
    }},
    ...,
    {{
        "question": "How does cg-noise generate variations in images?",
        "answer": "cg-noise generates variations in images by using a weight `x` and two seeds. It generates the noise based on `random_based_on(variation_seed) * x + random_based_on(seed) * (1-x)`."
    }}
]
#############

The file currently being passed in is about {0}, the specific contents are as follows: {1}
'''

# 从comfyui-nodes-docs[https://github.com/CavinHuang/comfyui-nodes-docs]使用LLMs构建问答数据的提示词模板,中文模板
system_prompt_zh = "我希望你扮演一个问答数据构建者的角色，根据我传入的文本，生成合理的问答数据对。不要编造传入文本中没有的信息。你需要根据传入文本的长度调整生成的问答数据对的数量，但每次至少生成 7 个问答数据对。"

template_zh = '''
# 上下文 #
我想对大型语言模型进行微调。我需要构建一个微调数据集，这需要生成大量的问答数据对。
#############

# 目标 #
你需要理解我输入的文档内容，然后自己构造问答数据对，以json格式返回。这里有几点需要注意：
1. 我传递的文档都是关于ComfyUI（使用稳定扩散模型生成图片和视频的GUI）以及扩展其功能的自定义节点或插件。在构建问答数据时，一定要明确主题是针对ComfyUI还是针对其某个自定义节点或插件。问答数据中的主体必须带有节点或插件的具体名称，例如\"ComfyUI-Manager\"节点；不要只使用\"扩展\"或\"自定义节点\"作为主体，这并不能表示问题是关于节点或插件的具体名称。
2. 你需要根据传入的文本长度调整生成的问答数据对的数量，但每次至少生成七个问答数据对。
3. 注意，在传入具体文档内容之前，我会告诉你文本描述的具体主体，你可以在构建问答数据时直接使用它。确保构建的问答数据尽可能覆盖文本的所有内容。
4. 不要生成任何关于“Licenses”和“Commit Version”的问答对.
5. 不要对文档中的“以相对位置表示的图片或视频等对象”生成问答对。
6. 不要遗漏必要的符号，但不要添加不必要的符号。
7. 生成问答对时，不要只从特定主体开始，还要从特征和现象出发指向输入文本中主体得问题。例如，LoadImage节点可以加载图片，不要只生成“LoadImage节点可以做什么？”这样的问题，还要生成“哪些节点可以加载图片？”这样的问题。
8. 请确保输出的json数据格式正确。
#############

# 语气 #
专业、技术
#############

# 回复例子 #
[
    {{
        "question": "cg-noise是什么?",
        "answer": "cg-noise 是 ComfyUI 中的一个自定义节点，它取代了 KSampler 和 KSampler Advanced，允许初始噪声出现细微变化。"
    }},
    ...,
    {{
        "question": "cg-noise 如何在图像中产生变化？",
        "answer": "cg-noise 使用权重“x”和两个种子在图像中产生变化。它根据'random_based_on(variation_seed) * x + random_based_on(seed) * (1-x)'生成噪声。"
    }}
]
#############

传入的文档内容描述的主体是 {0}, 文档具体内容如下：{1}
'''