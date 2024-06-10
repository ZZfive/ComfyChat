

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
5. Do not miss necessary symbols, but do not add unnecessary symbols.
6. When generating question-answer pairs, do not just start from a specific subject, but also ask questions that point to the subject in the input text from the characteristics and phenomena. For example, the LoadImage node can load images, so do not just generate questions like "What can the LoadImage node do?", but also generate questions like "What nodes can load images?"
7. Please ensure that the output json data format is correct. 
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