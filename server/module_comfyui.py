import io
import os
import re
import sys
import uuid
import json
import random
import subprocess
from collections import OrderedDict, deque
from typing import Dict, Tuple, List, Callable

import pytoml
import requests
import websocket
import urllib.parse
import gradio as gr
from PIL import Image

'''
TODO
1，部分初始参数独立出来，便与后续统一从配置中读取
2，验证几个请求comfyui接口函数中都建立websocket连接的必要性
3，验证cache中设置module作为key必要性
4，简化后将功能进程到demo中

'''


class Option:
    def __init__(self, config_path: str) -> None:
        with open(config_path, encoding='utf8') as f:
            config = pytoml.load(f)["comfyui"]
        # comfyui服务相关配置
        self.comfyui_port = config["server"]["port"]
        self.comfyui_file = config["server"]["file"]
        self.server_address = "127.0.0.1:" + str(self.comfyui_port)

        # comfyui模块相关配置
        self.design_mode = config["module"]["design_mode"]
        self.design_mode = config["module"]["design_mode"]
        self.controlnet_num = config["module"]["controlnet_num"]
        self.controlnet_saveimage = config["module"]["controlnet_saveimage"]
        self.prompt = config["module"]["prompt"]
        self.negative_prompt = config["module"]["negative_prompt"]
        self.output_dir = config["module"]["output_dir"]

        if self.design_mode == 1:
            self.width = 64
            self.hight = 64
            self.steps = 2
        else:
            self.width = 512
            self.hight = 768
            self.steps = 20

        self.client_id = str(uuid.uuid4())
        self.uploaded_image = {}

        self.interrupt = False
        self.lora_initialized = False
        self.upscale_initialized = False
        self.upscale_model_initialized = False
        self.controlnet_initialized = False


# 获取前端页面展示需要的所有内容
class Choices:
    def __init__(self, opt: Option) -> None:
        ws = websocket.WebSocket()
        ws.connect("ws://{}/ws?clientId={}".format(opt.server_address, opt.client_id))  # 与comfyui中的websocket建立连接
        self.object_info = requests.get(url="http://{}/object_info".format(opt.server_address)).json()  # 获取comfyui中所有节点
        self.embedding = requests.get(url="http://{}/embeddings".format(opt.server_address)).json()  # 获取comfyui中所有embeddings
        ws.close()

        self.ckpt = []  # 基模型名称，用于前端展示
        self.ckpt_list = {}  # 基模型名称与具体路径列表
        self.ckpt_files = self.object_info["ImageOnlyCheckpointLoader"]["input"]["required"]["ckpt_name"][0]
        self.hidden_ckpt = ["stable_cascade_stage_c.safetensors", "stable_cascade_stage_b.safetensors",
                            "svd_xt_1_1.safetensors", "control_v11p_sd15_canny_fp16.safetensors",
                            "control_v11f1p_sd15_depth_fp16.safetensors", "control_v11p_sd15_openpose_fp16.safetensors"]
        for ckpt_file in self.ckpt_files:
            _, file = os.path.split(ckpt_file)
            if file not in self.hidden_ckpt:
                self.ckpt.append(file)
            self.ckpt_list[file] = ckpt_file
        self.ckpt = sorted(self.ckpt)

        self.controlnet_model = []  # controlnet模型名称，用于前端展示
        self.controlnet_model_list = {}  # controlnet模型名称与具体路径列表
        self.controlnet_files = self.object_info["ControlNetLoader"]["input"]["required"]["control_net_name"][0]
        for controlnet_file in self.controlnet_files:
            _, file = os.path.split(controlnet_file)
            self.controlnet_model.append(file)
            self.controlnet_model_list[file] = controlnet_file
        self.controlnet_model = sorted(self.controlnet_model)

        self.preprocessor = ["Canny"]  # controlnet预处理器列表
        if "AIO_Preprocessor" in self.object_info:
            self.preprocessor = ["none", "Canny", "CannyEdgePreprocessor", "DepthAnythingPreprocessor", "DWPreprocessor", "OpenposePreprocessor"]
            for preprocessor in sorted(self.object_info["AIO_Preprocessor"]["input"]["optional"]["preprocessor"][0]):
                if preprocessor not in self.preprocessor:
                    self.preprocessor.append(preprocessor)
        
        self.lora = self.object_info["LoraLoader"]["input"]["required"]["lora_name"][0]
        self.sampler = self.object_info["KSampler"]["input"]["required"]["sampler_name"][0]
        self.scheduler = self.object_info["KSampler"]["input"]["required"]["scheduler"][0]
        self.upscale_method = self.object_info["ImageScaleBy"]["input"]["required"]["upscale_method"][0]
        self.upscale_model = self.object_info["UpscaleModelLoader"]["input"]["required"]["model_name"][0]
        self.vae = ["Automatic"]
        for i in sorted(self.object_info["VAELoader"]["input"]["required"]["vae_name"][0]):
            self.vae.append(i)


opt = Option(config_path="/root/code/ComfyChat/server/config.ini")
choices = Choices(opt)


# 提示词调整
def format_prompt(prompt: str) -> str:
    prompt = re.sub(r"\s+,", ",", prompt)
    prompt = re.sub(r"\s+", " ", prompt)
    prompt = re.sub(",,+", ",", prompt)
    prompt = re.sub(",", ", ", prompt)
    prompt = re.sub(r"\s+", " ", prompt)
    prompt = re.sub(r"^,", "", prompt)
    prompt = re.sub(r"^ ", "", prompt)
    prompt = re.sub(r" $", "", prompt)
    prompt = re.sub(r",$", "", prompt)
    prompt = re.sub(": ", ":", prompt)
    return prompt


 # 设置基模型
def get_model_path(ckpt_list: List[str], model_name: str) -> str:
    return ckpt_list[model_name]


# 设置随机种子
def gen_seed(seed: int):
    seed = int(seed)
    if seed < 0:
        seed = random.randint(0, 18446744073709551615)
    if seed > 18446744073709551615:
        seed = 18446744073709551615
    return seed


# 上传图片到comfyui
def upload_image(opt: Option, image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="png")
    image = buffer.getbuffer()
    image_hash = hash(image.tobytes())
    if image_hash in opt.uploaded_image:  # 通过哈希值防止图片重复上传
        return opt.uploaded_image[image_hash]
    image_name = str(uuid.uuid4()) + ".png"
    opt.uploaded_image[image_hash] = image_name
    image_file = {"image": (image_name, image)}

    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(opt.server_address, opt.client_id))
    requests.post(url="http://{}/upload/image".format(opt.server_address), files=image_file)
    ws.close()

    return image_name


# 构建工作流
def order_workflow(workflow: Dict):
    link_list = {}
    for node in workflow:
        node_link = []
        for input in workflow[node]["inputs"]:
            if isinstance(workflow[node]["inputs"][input], list):
                node_link.append(workflow[node]["inputs"][input][0])
        link_list[node] = node_link
    in_degree = {v: 0 for v in link_list}
    for node in link_list:
        for neighbor in link_list[node]:
            in_degree[neighbor] += 1
    queue = deque([node for node in in_degree if in_degree[node] == 0])
    order_list = []
    while queue:
        node = queue.popleft()
        order_list.append(node)
        for neighbor in link_list[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    order_list = order_list[::-1]
    max_nodes = 1000
    new_node_id = max_nodes * 10 + 1
    workflow_string = json.dumps(workflow)
    for node in order_list:
        workflow_string = workflow_string.replace(f'"{node}"', f'"{new_node_id}"')
        new_node_id += 1
    workflow = json.loads(workflow_string)
    workflow = OrderedDict(sorted(workflow.items()))
    new_node_id = 1
    workflow_string = json.dumps(workflow)
    for node in workflow:
        workflow_string = workflow_string.replace(f'"{node}"', f'"{new_node_id}"')
        new_node_id += 1
    workflow = json.loads(workflow_string)
    for node in workflow:
        if "_meta" in workflow[node]:
            del workflow[node]["_meta"]
    return workflow


# 中断任务
def post_interrupt(opt: Option) -> None:
    opt.interrupt = True
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(opt.server_address, opt.client_id))
    requests.post(url="http://{}/interrupt".format(opt.server_address))
    ws.close()


# 给负向提示词中添加选择的embeddings
def add_embedding(choices: Choices, embedding: List[str], negative_prompt: str) -> str:
    for embed in choices.embedding:
        negative_prompt = negative_prompt.replace(f"embedding:{embed},", "")  # 先去除字符串中可能已有的embedding

    negative_prompt = format_prompt(negative_prompt)
    for embed in embedding[::-1]:
        negative_prompt = f"embedding:{embed}, {negative_prompt}"
    
    return negative_prompt


# 基于workflow请求comfyui生成图片
def gen_image(opt: Option, workflow: Dict, counter: int, batch_count: int, progress: Callable) -> Tuple[List[Image.Image], str]:
    if counter == 1:
        progress(0, desc="Processing...")
    if batch_count == 1:
        batch_info = ""
    else:
        batch_info = f"Batch {counter}/{batch_count}: "
    workflow = order_workflow(workflow)
    current_progress = 0
    opt.interrupt = False

    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(opt.server_address, opt.client_id))
    data = {"prompt": workflow, "client_id": opt.client_id}
    prompt_id = requests.post(url="http://{}/prompt".format(opt.server_address), json=data).json()["prompt_id"]  # 请求下发任务

    while True:  # 下发请求后，comfyui在处理任务时会通过websocket发送信息，故在死循环中判断处理完毕信息，获取结果
        try:
            ws.settimeout(0.1)
            wsrecv = ws.recv()  # 接受comfyui返回的任务处理过程中的信息
            if isinstance(wsrecv, str):
                data = json.loads(wsrecv)["data"]
                if "node" in data:
                    if data["node"] is not None:
                        if "value" in data and "max" in data:
                            if data["max"] > 1:
                                current_progress = data["value"] / data["max"]  # 当前进度
                            progress(current_progress, desc=f"{batch_info}" + workflow[data["node"]]["class_type"] + " " + str(data["value"]) + "/" + str(data["max"]))
                        else:
                            progress(current_progress, desc=f"{batch_info}" + workflow[data["node"]]["class_type"])
                    if data["node"] is None and data["prompt_id"] == prompt_id:
                        break  # 任务处理结束
            else:
                continue
        except websocket.WebSocketTimeoutException:
            if opt.interrupt is True:
                ws.close()
                return None, None

    # 请求comfyui历史信息接口，基于prompt_id获取结果
    history = requests.get(url="http://{}/history/{}".format(opt.server_address, prompt_id)).json()[prompt_id]

    images = []
    file_path = ""
    for node_id in history["outputs"]:
        node_output = history["outputs"][node_id]
        if "images" in node_output:
            for image in node_output["images"]:
                file_path = opt.output_dir + image["filename"]
                data = {"filename": image["filename"], "subfolder": image["subfolder"], "type": image["type"]}
                url_values = urllib.parse.urlencode(data)
                # 请求comfyui的view接口获取图像
                image_data = requests.get("http://{}/view?{}".format(opt.server_address, url_values)).content
                image = Image.open(io.BytesIO(image_data))
                images.append(image)
    ws.close()
    return images, file_path


# 选择图片
def get_gallery_index(evt: gr.SelectData) -> int | tuple[int, int]:
    return evt.index


# 获取图片中的工作流
def get_image_info(image_pil: Image.Image) -> List | str:
    image_info=[]
    if image_pil is None:
        return
    for key, value in image_pil.info.items():
        image_info.append(value)
    if image_info != []:
        image_info = image_info[0]
        if image_info == 0:
            image_info = "None"
    else:
        image_info = "None"
    return image_info


# 将图片发送至别的tab
def send_to(data: List[Image.Image], index: int) -> Image.Image | None:
    if data == [] or data is None:
        return None
    return data[index]


class Lora:
    def __init__(self) -> None:
        self.cache = {}
        # self.initialized = False
 
    def add_node(self, module: str, workflow: Dict, node_id: int, model_port: int, clip_port: int) -> Tuple[Dict, int, int, int]:  # 构建该类节点适用于workflow的结构化对象
        for lora in self.cache[module]:
            strength_model = self.cache[module][lora]
            strength_clip = self.cache[module][lora]
            node_id += 1
            workflow[str(node_id)] = {"inputs": {"lora_name": lora, "strength_model": strength_model,
                                                 "strength_clip": strength_clip, "model": model_port,
                                                 "clip": clip_port}, "class_type": "LoraLoader"}
            model_port = [str(node_id), 0]
            clip_port = [str(node_id), 1]
        return workflow, node_id, model_port, clip_port
 
    def update_cache(self, module: str, lora: str, lora_weight: str):
        # if self.initialized is False:
        #     self.cache = {}

        if lora == []:
            self.cache[module] = {}
            return True, [], gr.update(value="", visible=False)
        
        # lora_weight中是已经构建好的lora特定格式的字符串
        lora_list = {}
        for i in lora_weight.split("<"):
            for j in i.split(">"):
                if j != "" and ":" in j:
                    lora_name, weight = j.split(":")
                    lora_list[lora_name] = weight

        lora_weight = ""
        self.cache[module] = {}  # lora发生改变时，Lora.cache[module]都会重新赋值
        for i in lora:
            if i in lora_list:
                weight = lora_list[i]
            else:
                weight = opt.lora_weight
            if lora.index(i) == 0:
                lora_weight = f"<{i}:{weight}>"
            else:
                lora_weight = f"{lora_weight}\n\n<{i}:{weight}>"
            if weight != "":
                weight = float(weight)
            self.cache[module][i] = weight
        return gr.update(), gr.update(value=lora_weight, visible=True)
 
    def blocks(self, module: str, choices: Choices) -> None:
        module = gr.Textbox(value=module, visible=False)  # 作为Lora.cache中的一个key
        lora = gr.Dropdown(choices.lora, label="Lora", multiselect=True, interactive=True)
        lora_weight = gr.Textbox(label="Lora weight | Lora 权重", visible=False)
        for gr_block in [lora, lora_weight]:  # 感觉此处对lora_weight改变为用，其就是基于lora来改变的  TODO 待验证
            gr_block.change(fn=self.update_cache, inputs=[module, lora, lora_weight], outputs=[lora, lora_weight])


class Upscale:
    def __init__(self) -> None:
        self.cache = {}
 
    def add_node(self, module: str, workflow: Dict, node_id: int, image_port: int) -> Tuple[Dict, int, int]:
        upscale_method = self.cache[module]["upscale_method"]
        scale_by = self.cache[module]["scale_by"]
        node_id += 1
        workflow[str(node_id)] = {"inputs": {"upscale_method": upscale_method, "scale_by": scale_by, "image": image_port}, "class_type": "ImageScaleBy"}
        image_port = [str(node_id), 0]
        return workflow, node_id, image_port
 
    def auto_enable(self, scale_by: int):
        if scale_by > 1:
            return True
        else:
            return False
 
    def update_cache(self, module: str, enable: bool, upscale_method: str, scale_by: int) -> None:
        if module not in self.cache:
            self.cache[module] = {}

        if enable is True:
            self.cache[module]["upscale_method"] = upscale_method
            self.cache[module]["scale_by"] = scale_by
        else:
            del self.cache[module]
 
    def blocks(self, module: str, choices: Choices) -> None:
        module = gr.Textbox(value=module, visible=False)
        enable = gr.Checkbox(label="Enable（放大系数大于1后自动启用）")

        with gr.Row():
            upscale_method = gr.Dropdown(choices.upscale_method, label="Upscale method | 放大方法", value=choices.upscale_method[-1])
            scale_by = gr.Slider(minimum=1, maximum=8, step=1, label="Scale by | 放大系数", value=1)

        scale_by.release(fn=self.auto_enable, inputs=[scale_by], outputs=[enable])
        inputs = [module, enable, upscale_method, scale_by]

        for gr_block in inputs:
            if type(gr_block) is gr.components.slider.Slider:
                gr_block.release(fn=self.update_cache, inputs=inputs)
            else:
                gr_block.change(fn=self.update_cache, inputs=inputs)


class UpscaleWithModel:
    def __init__(self) -> None:
        self.cache = {}
 
    def add_node(self, module: str, workflow: Dict, node_id: int, image_port: int) -> Tuple[Dict, int, int]:
        upscale_model = self.cache[module]["upscale_model"]
        node_id += 1
        workflow[str(node_id)] = {"inputs": {"model_name": upscale_model}, "class_type": "UpscaleModelLoader"}
        upscale_model_port = [str(node_id), 0]
        node_id += 1
        workflow[str(node_id)] = {"inputs": {"upscale_model": upscale_model_port, "image": image_port}, "class_type": "ImageUpscaleWithModel"}
        image_port = [str(node_id), 0]
        return workflow, node_id, image_port
 
    def update_cache(self, module: str, enable: bool, upscale_model: str) -> None:
        if module not in self.cache:
            self.cache[module] = {}
        if enable is True:
            self.cache[module]["upscale_model"] = upscale_model
        else:
            del self.cache[module]
 
    def blocks(self, module: str, choices: Choices) -> None:
        module = gr.Textbox(value=module, visible=False)
        enable = gr.Checkbox(label="Enable")
        upscale_model = gr.Dropdown(choices.upscale_model, label="Upscale model | 超分模型", value=choices.upscale_model[0])
        inputs = [module, enable, upscale_model]
        for gr_block in inputs:
            gr_block.change(fn=self.update_cache, inputs=inputs)


class ControlNet:
    def __init__(self, opt: Option) -> None:
        self.cache = {}
        self.opt = opt

        self.model_preprocessor_list = {
            "control_v11e_sd15_ip2p.safetensors": [],
            "control_v11e_sd15_shuffle.safetensors": ["ShufflePreprocessor"],
            "control_v11f1e_sd15_tile.bin": ["TilePreprocessor", "TTPlanet_TileGF_Preprocessor", "TTPlanet_TileSimple_Preprocessor"],
            "control_v11f1p_sd15_depth.safetensors": ["DepthAnythingPreprocessor", "LeReS-DepthMapPreprocessor", "MiDaS-NormalMapPreprocessor", "MeshGraphormer-DepthMapPreprocessor", "MeshGraphormer+ImpactDetector-DepthMapPreprocessor", "MiDaS-DepthMapPreprocessor", "Zoe_DepthAnythingPreprocessor", "Zoe-DepthMapPreprocessor"],
            "control_v11p_sd15_canny.safetensors": ["Canny", "CannyEdgePreprocessor"],
            "control_v11p_sd15_inpaint.safetensors": [],
            "control_v11p_sd15_lineart.safetensors": ["LineArtPreprocessor", "LineartStandardPreprocessor"],
            "control_v11p_sd15_mlsd.safetensors": ["M-LSDPreprocessor"],
            "control_v11p_sd15_normalbae.safetensors": ["BAE-NormalMapPreprocessor", "DSINE-NormalMapPreprocessor"],
            "control_v11p_sd15_openpose.safetensors": ["DWPreprocessor", "OpenposePreprocessor", "DensePosePreprocessor"],
            "control_v11p_sd15_scribble.safetensors": ["ScribblePreprocessor", "Scribble_XDoG_Preprocessor", "Scribble_PiDiNet_Preprocessor", "FakeScribblePreprocessor"],
            "control_v11p_sd15_seg.safetensors": ["AnimeFace_SemSegPreprocessor", "OneFormer-COCO-SemSegPreprocessor", "OneFormer-ADE20K-SemSegPreprocessor", "SemSegPreprocessor", "UniFormer-SemSegPreprocessor"],
            "control_v11p_sd15_softedge.safetensors": ["HEDPreprocessor", "PiDiNetPreprocessor", "TEEDPreprocessor", "DiffusionEdge_Preprocessor"],
            "control_v11p_sd15s2_lineart_anime.safetensors": ["AnimeLineArtPreprocessor", "Manga2Anime_LineArt_Preprocessor"],
            "control_scribble.safetensors": ["BinaryPreprocessor"],
            "ioclab_sd15_recolor.safetensors": ["ImageLuminanceDetector", "ImageIntensityDetector"],
            "control_sd15_animal_openpose_fp16.pth": ["AnimalPosePreprocessor"],
            "controlnet_sd21_laion_face_v2.safetensors": ["MediaPipe-FaceMeshPreprocessor"]
        }
 
    def add_node(self, module: str, counter: int, workflow: Dict, node_id: int, positive_port: int, negative_port: int) -> Tuple[Dict, int, int, int]:
        for unit_id in self.cache[module]:
            preprocessor = self.cache[module][unit_id]["preprocessor"]
            model = self.cache[module][unit_id]["model"]
            input_image = self.cache[module][unit_id]["input_image"]
            resolution = self.cache[module][unit_id]["resolution"]
            strength = self.cache[module][unit_id]["strength"]
            start_percent = self.cache[module][unit_id]["start_percent"]
            end_percent = self.cache[module][unit_id]["end_percent"]
            node_id += 1
            workflow[str(node_id)] = {"inputs": {"image": input_image, "upload": "image"}, "class_type": "LoadImage"}
            image_port = [str(node_id), 0]

            if preprocessor == "Canny":
                node_id += 1
                workflow[str(node_id)] = {"inputs": {"low_threshold": 0.3, "high_threshold": 0.7, "image": image_port}, "class_type": "Canny"}
                image_port = [str(node_id), 0]
            else:
                node_id += 1
                workflow[str(node_id)] = {"inputs": {"preprocessor": preprocessor, "resolution": resolution, "image": image_port}, "class_type": "AIO_Preprocessor"}
                image_port = [str(node_id), 0]
            if counter == 1 and self.controlnet_saveimage == 1:
                node_id += 1
                workflow[str(node_id)] = {"inputs": {"filename_prefix": "ControlNet", "images": image_port}, "class_type": "SaveImage"}
            node_id += 1
            workflow[str(node_id)] = {"inputs": {"control_net_name": model}, "class_type": "ControlNetLoader"}
            control_net_port = [str(node_id), 0]
            node_id += 1
            workflow[str(node_id)] = {"inputs": {"strength": strength, "start_percent": start_percent, "end_percent": end_percent, "positive": positive_port, "negative": negative_port, "control_net": control_net_port, "image": image_port}, "class_type": "ControlNetApplyAdvanced"}
            positive_port = [str(node_id), 0]
            negative_port = [str(node_id), 1]
        
        return workflow, node_id, positive_port, negative_port
 
    def auto_enable(self) -> bool:
        return True
    
    def auto_select_model(self, preprocessor: str, choices: Choices) -> str:
        for model in choices.controlnet_model:
            if model in self.model_preprocessor_list:
                if preprocessor in self.model_preprocessor_list[model]:
                    return gr.update(value=model)
        return gr.update(value="未定义/检测到对应的模型，请自行选择！")
 
    def preprocess(self, unit_id: int, preview: bool, preprocessor: str, input_image: Image.Image,
                   resolution: int, progress: Callable = gr.Progress()) -> None | List[Image.Image]:
        if preview is False or input_image is None:
            return
        input_image = upload_image(self.opt, input_image)
        workflow = {}
        node_id = 1
        workflow[str(node_id)] = {"inputs": {"image": input_image, "upload": "image"}, "class_type": "LoadImage"}
        image_port = [str(node_id), 0]
        if preprocessor == "Canny":
            node_id += 1
            workflow[str(node_id)] = {"inputs": {"low_threshold": 0.3, "high_threshold": 0.7, "image": image_port}, "class_type": "Canny"}
            image_port = [str(node_id), 0]
        else:
            node_id += 1
            workflow[str(node_id)] = {"inputs": {"preprocessor": preprocessor, "resolution": resolution, "image": image_port}, "class_type": "AIO_Preprocessor"}
            image_port = [str(node_id), 0]
        node_id += 1
        workflow[str(node_id)] = {"inputs": {"images": image_port}, "class_type": "PreviewImage"}
        output = gen_image(workflow, 1, 1, progress)[0]
        if output is not None:
            output = output[0]
        return output
 
    def update_cache(self, module: str, unit_id: int, enable: bool, preprocessor: str, model: str, input_image: Image.Image,
                     resolution: int, strength: float, start_percent: float, end_percent: float, choices: Choices) -> bool:
        if module not in self.cache:
            self.cache[module] = {}

        self.cache[module][unit_id] = {}

        if input_image is None:
            del self.cache[module][unit_id]
            return False
        
        if model not in choices.controlnet_model:
            del self.cache[module][unit_id]
            return False
        
        if enable is True:
            self.cache[module][unit_id]["preprocessor"] = preprocessor
            self.cache[module][unit_id]["model"] = choices.controlnet_model_list[model]
            self.cache[module][unit_id]["input_image"] = upload_image(input_image)
            self.cache[module][unit_id]["resolution"] = resolution
            self.cache[module][unit_id]["strength"] = strength
            self.cache[module][unit_id]["start_percent"] = start_percent
            self.cache[module][unit_id]["end_percent"] = end_percent
        else:
            del self.cache[module][unit_id]
        
        return gr.update()
 
    def unit(self, module: str, i: int, choices: Choices) -> None:
        module = gr.Textbox(value=module, visible=False)
        unit_id = gr.Textbox(value=i, visible=False)
        with gr.Row():
            enable = gr.Checkbox(label="Enable（上传图片后自动启用）")
            preview = gr.Checkbox(label="Preview")
        with gr.Row():
            preprocessor = gr.Dropdown(choices.preprocessor, label="Preprocessor", value="Canny")
            model = gr.Dropdown(choices.controlnet_model, label="ControlNet model", value="control_v11p_sd15_canny.safetensors")
        with gr.Row():
            input_image = gr.Image(type="pil")
            preprocess_preview = gr.Image(label="Preprocessor preview")
        with gr.Row():
            resolution = gr.Slider(label="Resolution", minimum=64, maximum=2048, step=64, value=512)
            strength = gr.Slider(label="Strength", minimum=0, maximum=2, step=0.01, value=1)
        with gr.Row():
            start_percent = gr.Slider(label="Start percent", minimum=0, maximum=1, step=0.01, value=0)
            end_percent = gr.Slider(label="End percent", minimum=0, maximum=1, step=0.01, value=1)
        input_image.upload(fn=self.auto_enable, inputs=None, outputs=[enable])
        preprocessor.change(fn=self.auto_select_model, inputs=[preprocessor], outputs=[model])
        for gr_block in [preview, preprocessor, input_image]:
            gr_block.change(fn=self.preprocess, inputs=[unit_id, preview, preprocessor, input_image, resolution], outputs=[preprocess_preview])
        inputs = [module, unit_id, enable, preprocessor, model, input_image, resolution, strength, start_percent, end_percent]
        for gr_block in inputs:
            if type(gr_block) is gr.components.slider.Slider:
                gr_block.release(fn=self.update_cache, inputs=inputs, outputs=enable)
            else:
                gr_block.change(fn=self.update_cache, inputs=inputs, outputs=enable)
 
    def blocks(self, module):
        with gr.Tab(label="控制网络"):
            if self.opt.controlnet_num == 1:
                self.unit(module, 1)
            else:
                for i in range(self.opt.controlnet_num):
                    with gr.Tab(label=f"ControlNet Unit {i + 1}"):
                        self.unit(module, i + 1)


class Postprocess:
    def __init__(self, upscale: Upscale, upscal_model: UpscaleWithModel) -> None:
        self.upscale = upscale
        self.upscal_model = upscal_model

    def add_node(self, module: str, *args) -> Tuple[Dict, int, int]:
        if module == "SD":
            workflow, node_id, image_port, model_port, clip_port, vae_port, positive_port, negative_port, seed, steps, cfg, sampler_name, scheduler = args
        else:
            workflow, node_id, image_port = args
        
        if module in self.upscale.cache:
            workflow, node_id, image_port = self.upscale.add_node(module, workflow, node_id, image_port)

        if module in self.upscal_model.cache:
            workflow, node_id, image_port = self.upscal_model.add_node(module, workflow, node_id, image_port)

        return workflow, node_id, image_port
 
    def blocks(self, module: str) -> None:
        with gr.Tab(label="图像放大"):
            with gr.Row():
                with gr.Tab(label="算术放大"):
                    Upscale.blocks(module)
            with gr.Row():
                with gr.Tab(label="超分放大"):
                    UpscaleWithModel.blocks(module)
            gr.HTML("注意：同时启用两种放大模式将先执行算术放大，再执行超分放大，最终放大倍数为二者放大倍数的乘积！")


class SD:
    def generate(initialized, batch_count, ckpt_name, vae_name, clip_mode, clip_skip, width, height, batch_size, negative_prompt, positive_prompt, seed, steps, cfg, sampler_name, scheduler, denoise, input_image, progress=gr.Progress()):
        module = "SD"
        ckpt_name = Function.get_model_path(ckpt_name)
        seed = Function.gen_seed(seed)

        if input_image is not None:
            input_image = Function.upload_image(input_image)

        counter = 1
        output_images = []
        node_id = 0

        while counter <= batch_count:
            workflow = {}
            node_id += 1
            workflow[str(node_id)] = {"inputs": {"ckpt_name": ckpt_name}, "class_type": "CheckpointLoaderSimple"}
            model_port = [str(node_id), 0]
            clip_port = [str(node_id), 1]
            vae_port = [str(node_id), 2]
            if vae_name != "Automatic":
                node_id += 1
                workflow[str(node_id)] = {"inputs": {"vae_name": vae_name}, "class_type": "VAELoader"}
                vae_port = [str(node_id), 0]
            if input_image is None:
                node_id += 1
                workflow[str(node_id)] = {"inputs": {"width": width, "height": height, "batch_size": batch_size}, "class_type": "EmptyLatentImage"}
                latent_image_port = [str(node_id), 0]
            else:
                node_id += 1
                workflow[str(node_id)] = {"inputs": {"image": input_image, "upload": "image"}, "class_type": "LoadImage"}
                pixels_port = [str(node_id), 0]
                node_id += 1
                workflow[str(node_id)] = {"inputs": {"pixels": pixels_port, "vae": vae_port}, "class_type": "VAEEncode"}
                latent_image_port = [str(node_id), 0]
            node_id += 1
            workflow[str(node_id)] = {"inputs": {"stop_at_clip_layer": -clip_skip, "clip": clip_port}, "class_type": "CLIPSetLastLayer"}
            clip_port = [str(node_id), 0]
            if initialized is True and module in Lora.cache:
                workflow, node_id, model_port, clip_port = Lora.add_node(module, workflow, node_id, model_port, clip_port)
            node_id += 1
            if clip_mode == "ComfyUI":
                workflow[str(node_id)] = {"inputs": {"text": positive_prompt, "clip": clip_port}, "class_type": "CLIPTextEncode"}
            else:
                workflow[str(node_id)] = {"inputs": {"text": positive_prompt, "token_normalization": "none", "weight_interpretation": "A1111", "clip": clip_port}, "class_type": "BNK_CLIPTextEncodeAdvanced"}
            positive_port = [str(node_id), 0]
            node_id += 1
            if clip_mode == "ComfyUI":
                workflow[str(node_id)] = {"inputs": {"text": negative_prompt, "clip": clip_port}, "class_type": "CLIPTextEncode"}
            else:
                workflow[str(node_id)] = {"inputs": {"text": negative_prompt, "token_normalization": "none", "weight_interpretation": "A1111", "clip": clip_port}, "class_type": "BNK_CLIPTextEncodeAdvanced"}
            negative_port = [str(node_id), 0]
            if initialized is True and module in ControlNet.cache:
                workflow, node_id, positive_port, negative_port = ControlNet.add_node(module, counter, workflow, node_id, positive_port, negative_port)
            node_id += 1
            workflow[str(node_id)] = {"inputs": {"seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler_name, "scheduler": scheduler, "denoise": denoise, "model": model_port, "positive": positive_port, "negative": negative_port, "latent_image": latent_image_port}, "class_type": "KSampler"}
            samples_port = [str(node_id), 0]
            node_id += 1
            workflow[str(node_id)] = {"inputs": {"samples": samples_port, "vae": vae_port}, "class_type": "VAEDecode"}
            image_port = [str(node_id), 0]
            if initialized is True:
                workflow, node_id, image_port = Postprocess.add_node(module, workflow, node_id, image_port, model_port, clip_port, vae_port, positive_port, negative_port, seed, steps, cfg, sampler_name, scheduler)
            node_id += 1
            workflow[str(node_id)] = {"inputs": {"filename_prefix": "ComfyUI", "images": image_port}, "class_type": "SaveImage"}
            images = Function.gen_image(workflow, counter, batch_count, progress)[0]
            if images is None:
                break
            for image in images:
                output_images.append(image)
            seed += 1
            counter += 1
        return output_images, output_images
 
    def blocks():
        with gr.Row():
            with gr.Column():
                positive_prompt = gr.Textbox(placeholder="Positive prompt | 正向提示词", show_label=False, value=Default.prompt, lines=3)
                negative_prompt = gr.Textbox(placeholder="Negative prompt | 负向提示词", show_label=False, value=Default.negative_prompt, lines=3)
                with gr.Tab(label="基础设置"):
                    with gr.Row():
                        ckpt_name = gr.Dropdown(Choices.ckpt, label="Ckpt name | Ckpt 模型名称", value=Choices.ckpt[0])
                        vae_name = gr.Dropdown(Choices.vae, label="VAE name | VAE 模型名称", value=Choices.vae[0])
                        if "BNK_CLIPTextEncodeAdvanced" in Choices.object_info:
                            clip_mode = gr.Dropdown(["ComfyUI", "WebUI"], label="Clip 编码类型", value="ComfyUI")
                        else:
                            clip_mode = gr.Dropdown(["ComfyUI", "WebUI"], label="Clip 编码类型", value="ComfyUI", visible=False)
                        clip_skip = gr.Slider(minimum=1, maximum=12, step=1, label="Clip 跳过", value=1)
                    with gr.Row():
                        width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width | 图像宽度", value=Default.width)
                        batch_size = gr.Slider(minimum=1, maximum=8, step=1, label="Batch size | 批次大小", value=1)
                    with gr.Row():
                        height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height | 图像高度", value=Default.hight)
                        batch_count = gr.Slider(minimum=1, maximum=100, step=1, label="Batch count | 生成批次", value=1)
                    with gr.Row():
                        if Choices.lora != []:  # 当comfyui中有lora模型时前端界面才会初始对应模块
                            Lora.blocks("SD")
                        if Choices.embedding != []:
                            embedding = gr.Dropdown(Choices.embedding, label="Embedding", multiselect=True, interactive=True)
                            embedding.change(fn=Function.add_embedding, inputs=[embedding, negative_prompt], outputs=[negative_prompt])
                    with gr.Row():
                        SD.input_image = gr.Image(value=None, type="pil")
                        gr.HTML("<br>上传图片即自动转为图生图模式。<br><br>文生图、图生图模式共享设置参数。<br><br>图像宽度、图像高度、批次大小对图生图无效。")
                with gr.Tab(label="采样设置"):
                    with gr.Row():
                        sampler_name = gr.Dropdown(Choices.sampler, label="Sampling method | 采样方法", value=Choices.sampler[12])
                        scheduler = gr.Dropdown(Choices.scheduler, label="Schedule type | 采样计划表类型", value=Choices.scheduler[1])
                    with gr.Row():
                        denoise = gr.Slider(minimum=0, maximum=1, step=0.05, label="Denoise | 去噪强度", value=1)
                        steps = gr.Slider(minimum=1, maximum=100, step=1, label="Sampling steps | 采样次数", value=Default.steps)
                    with gr.Row():
                        cfg = gr.Slider(minimum=0, maximum=20, step=0.1, label="CFG Scale | CFG权重", value=7)
                        seed = gr.Slider(minimum=-1, maximum=18446744073709550000, step=1, label="Seed | 种子数", value=-1)
                if Choices.controlnet_model != []:
                    ControlNet.blocks("SD")
                Postprocess.blocks("SD")
            with gr.Column():
                with gr.Row():
                    btn = gr.Button("Generate | 生成", elem_id="button")
                    btn2 = gr.Button("Interrupt | 终止")
                output = gr.Gallery(preview=True, height=600)
                with gr.Row():
                    SD.send_to_sd = gr.Button("发送图片至 SD")
                    if SC.enable is True:
                        SD.send_to_sc = gr.Button("发送图片至 SC")
                    if SVD.enable is True:
                        SD.send_to_svd = gr.Button("发送图片至 SVD")
                    SD.send_to_extras = gr.Button("发送图片至 Extras")
                    SD.send_to_info = gr.Button("发送图片至 Info")
        SD.data = gr.State()
        SD.index = gr.State()
        btn.click(fn=SD.generate, inputs=[Initial.initialized, batch_count, ckpt_name, vae_name, clip_mode, clip_skip, width, height,
                                          batch_size, negative_prompt, positive_prompt, seed, steps, cfg, sampler_name, scheduler,
                                          denoise, SD.input_image], outputs=[output, SD.data])
        btn2.click(fn=Function.post_interrupt, inputs=None, outputs=None)
        output.select(fn=Function.get_gallery_index, inputs=None, outputs=[SD.index])


class SC:
    if Default.design_mode == 1:
        enable = True
    elif "stable_cascade_stage_c.safetensors" in Choices.ckpt_list and "stable_cascade_stage_b.safetensors" in Choices.ckpt_list:
        enable = True
    else:
        enable = False
 
    def generate(initialized, batch_count, positive_prompt, negative_prompt, width, height, batch_size, seed_c, steps_c, cfg_c, sampler_name_c, scheduler_c, denoise_c, seed_b, steps_b, cfg_b, sampler_name_b, scheduler_b, denoise_b, input_image, progress=gr.Progress()):
        module = "SC"
        ckpt_name_c = Function.get_model_path("stable_cascade_stage_c.safetensors")
        ckpt_name_b = Function.get_model_path("stable_cascade_stage_b.safetensors")
        seed_c = Function.gen_seed(seed_c)
        seed_b = Function.gen_seed(seed_b)
        if input_image is not None:
            input_image = Function.upload_image(input_image)
        counter = 1
        output_images = []
        while counter <= batch_count:
            workflow = {
"1": {"inputs": {"ckpt_name": ckpt_name_c}, "class_type": "CheckpointLoaderSimple"},
"2": {"inputs": {"image": input_image, "upload": "image"}, "class_type": "LoadImage"},
"3": {"inputs": {"compression": 42, "image": ["2", 0], "vae": ["1", 2]}, "class_type": "StableCascade_StageC_VAEEncode"},
"4": {"inputs": {"text": negative_prompt, "clip": ["1", 1]}, "class_type": "CLIPTextEncode"},
"5": {"inputs": {"text": positive_prompt, "clip": ["1", 1]}, "class_type": "CLIPTextEncode"},
"6": {"inputs": {"seed": seed_c, "steps": steps_c, "cfg": cfg_c, "sampler_name": sampler_name_c, "scheduler": scheduler_c, "denoise": denoise_c, "model": ["1", 0], "positive": ["5", 0], "negative": ["4", 0], "latent_image": ["3", 0]}, "class_type": "KSampler"},
"7": {"inputs": {"conditioning": ["5", 0], "stage_c": ["6", 0]}, "class_type": "StableCascade_StageB_Conditioning"},
"8": {"inputs": {"ckpt_name": ckpt_name_b}, "class_type": "CheckpointLoaderSimple"},
"9": {"inputs": {"seed": seed_b, "steps": steps_b, "cfg": cfg_b, "sampler_name": sampler_name_b, "scheduler": scheduler_b, "denoise": denoise_b, "model": ["8", 0], "positive": ["7", 0], "negative": ["4", 0], "latent_image": ["3", 1]}, "class_type": "KSampler"},
"10": {"inputs": {"samples": ["9", 0], "vae": ["8", 2]}, "class_type": "VAEDecode"}
}
            if input_image is None:
                del workflow["2"]
                workflow["3"] = {"inputs": {"width": width, "height": height, "compression": 42, "batch_size": batch_size}, "class_type": "StableCascade_EmptyLatentImage"}
            node_id = 10
            image_port = [str(node_id), 0]
            if initialized is True:
                workflow, node_id, image_port = Postprocess.add_node(module, workflow, node_id, image_port)
            node_id += 1
            workflow[str(node_id)] = {"inputs": {"filename_prefix": "ComfyUI", "images": image_port}, "class_type": "SaveImage"}
            images = Function.gen_image(workflow, counter, batch_count, progress)[0]
            if images is None:
                break
            for image in images:
                output_images.append(image)
            seed_c += 1
            counter += 1
        return output_images, output_images
 
    def blocks():
        with gr.Row():
            with gr.Column():
                positive_prompt = gr.Textbox(placeholder="Positive prompt | 正向提示词", show_label=False, value=Default.prompt, lines=3)
                negative_prompt = gr.Textbox(placeholder="Negative prompt | 负向提示词", show_label=False, value=Default.negative_prompt, lines=3)
                with gr.Tab(label="基础设置"):
                    with gr.Row():
                        width = gr.Slider(minimum=128, maximum=2048, step=128, label="Width | 图像宽度", value=1024)
                        batch_size = gr.Slider(minimum=1, maximum=8, step=1, label="Batch size | 批次大小", value=1)
                    with gr.Row():
                        height = gr.Slider(minimum=128, maximum=2048, step=128, label="Height | 图像高度", value=1024)
                        batch_count = gr.Slider(minimum=1, maximum=100, step=1, label="Batch count | 生成批次", value=1)
                    with gr.Row():
                        SC.input_image = gr.Image(value=None, type="pil")
                        gr.HTML("<br>上传图片即自动转为图生图模式。<br><br>文生图、图生图模式共享设置参数。<br><br>图像宽度、图像高度、批次大小对图生图无效。")
                with gr.Tab(label="Stage C 采样设置"):
                    with gr.Row():
                        sampler_name_c = gr.Dropdown(Choices.sampler, label="Sampling method | 采样方法", value=Choices.sampler[12])
                        scheduler_c = gr.Dropdown(Choices.scheduler, label="Schedule type | 采样计划表类型", value=Choices.scheduler[1])
                    with gr.Row():
                        denoise_c = gr.Slider(minimum=0, maximum=1, step=0.05, label="Denoise | 去噪强度", value=1)
                        steps_c = gr.Slider(minimum=10, maximum=30, step=1, label="Sampling steps | 采样次数", value=20)
                    with gr.Row():
                        cfg_c = gr.Slider(minimum=0, maximum=20, step=0.1, label="CFG Scale | CFG权重", value=4)
                        seed_c = gr.Slider(minimum=-1, maximum=18446744073709550000, step=1, label="Seed | 种子数", value=-1)
                with gr.Tab(label="Stage B 采样设置"):
                    with gr.Row():
                        sampler_name_b = gr.Dropdown(Choices.sampler, label="Sampling method | 采样方法", value=Choices.sampler[12])
                        scheduler_b = gr.Dropdown(Choices.scheduler, label="Schedule type | 采样计划表类型", value=Choices.scheduler[1])
                    with gr.Row():
                        denoise_b = gr.Slider(minimum=0, maximum=1, step=0.05, label="Denoise | 去噪强度", value=1)
                        steps_b = gr.Slider(minimum=4, maximum=12, step=1, label="Sampling steps | 采样次数", value=10)
                    with gr.Row():
                        cfg_b = gr.Slider(minimum=0, maximum=20, step=0.1, label="CFG Scale | CFG权重", value=1.1)
                        seed_b = gr.Slider(minimum=-1, maximum=18446744073709550000, step=1, label="Seed | 种子数", value=-1)
                Postprocess.blocks("SC")
            with gr.Column():
                with gr.Row():
                    btn = gr.Button("Generate | 生成", elem_id="button")
                    btn2 = gr.Button("Interrupt | 终止")
                output = gr.Gallery(preview=True, height=600)
                with gr.Row():
                    SC.send_to_sd = gr.Button("发送图片至 SD")
                    SC.send_to_sc = gr.Button("发送图片至 SC")
                    if SVD.enable is True:
                        SC.send_to_svd = gr.Button("发送图片至 SVD")
                    SC.send_to_extras = gr.Button("发送图片至 Extras")
                    SC.send_to_info = gr.Button("发送图片至 Info")
        SC.data = gr.State()
        SC.index = gr.State()
        btn.click(fn=SC.generate, inputs=[Initial.initialized, batch_count, positive_prompt, negative_prompt, width, height, batch_size, seed_c, steps_c, cfg_c, sampler_name_c, scheduler_c, denoise_c, seed_b, steps_b, cfg_b, sampler_name_b, scheduler_b, denoise_b, SC.input_image], outputs=[output, SC.data])
        btn2.click(fn=Function.post_interrupt, inputs=None, outputs=None)
        output.select(fn=Function.get_gallery_index, inputs=None, outputs=[SC.index])


class SVD:
    if Default.design_mode == 1:
        enable = True
    elif "svd_xt_1_1.safetensors" in Choices.ckpt_list:
        enable = True
    else:
        enable = False
 
    def generate(input_image, width, height, video_frames, motion_bucket_id, fps, augmentation_level, min_cfg, seed, steps, cfg, sampler_name, scheduler, denoise, fps2, lossless, quality, method, progress=gr.Progress()):
        ckpt_name = Function.get_model_path("svd_xt_1_1.safetensors")
        seed = Function.gen_seed(seed)
        if input_image is None:
            return
        else:
            input_image = Function.upload_image(input_image)
        workflow = {
"1": {"inputs": {"ckpt_name": ckpt_name}, "class_type": "ImageOnlyCheckpointLoader"},
"2": {"inputs": {"image": input_image, "upload": "image"}, "class_type": "LoadImage"},
"3": {"inputs": {"width": width, "height": height, "video_frames": video_frames, "motion_bucket_id": motion_bucket_id, "fps": fps, "augmentation_level": augmentation_level, "clip_vision": ["1", 1], "init_image": ["2", 0], "vae": ["1", 2]}, "class_type": "SVD_img2vid_Conditioning"},
"4": {"inputs": {"min_cfg": min_cfg, "model": ["1", 0]}, "class_type": "VideoLinearCFGGuidance"},
"5": {"inputs": {"seed": seed, "steps": steps, "cfg": cfg, "sampler_name": sampler_name, "scheduler": scheduler, "denoise": denoise, "model": ["4", 0], "positive": ["3", 0], "negative": ["3", 1], "latent_image": ["3", 2]}, "class_type": "KSampler"},
"6": {"inputs": {"samples": ["5", 0], "vae": ["1", 2]}, "class_type": "VAEDecode"},
"7": {"inputs": {"filename_prefix": "ComfyUI", "fps": fps2, "lossless": False, "quality": quality, "method": method, "images": ["6", 0]}, "class_type": "SaveAnimatedWEBP"}
}
        return Function.gen_image(workflow, 1, 1, progress)[1]
 
    def blocks():
        with gr.Row():
            with gr.Column():
                SVD.input_image = gr.Image(value=None, type="pil")
                with gr.Tab(label="基础设置"):
                    with gr.Row():
                        width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width | 图像宽度", value=512)
                        video_frames = gr.Slider(minimum=1, maximum=25, step=1, label="Video frames | 视频帧", value=25)
                    with gr.Row():
                        height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height | 图像高度", value=512)
                        fps = gr.Slider(minimum=1, maximum=30, step=1, label="FPS | 帧率", value=6)
                    with gr.Row():
                        with gr.Column():
                            augmentation_level = gr.Slider(minimum=0, maximum=1, step=0.01, label="Augmentation level | 增强级别", value=0)
                            motion_bucket_id = gr.Slider(minimum=1, maximum=256, step=1, label="Motion bucket id | 运动参数", value=127)
                        with gr.Column():
                            min_cfg = gr.Slider(minimum=0, maximum=20, step=0.5, label="Min CFG | 最小CFG权重", value=1)
                with gr.Tab(label="采样设置"):
                    with gr.Row():
                        sampler_name = gr.Dropdown(Choices.sampler, label="Sampling method | 采样方法", value=Choices.sampler[12])
                        scheduler = gr.Dropdown(Choices.scheduler, label="Schedule type | 采样计划表类型", value=Choices.scheduler[1])
                    with gr.Row():
                        denoise = gr.Slider(minimum=0, maximum=1, step=0.05, label="Denoise | 去噪强度", value=1)
                        steps = gr.Slider(minimum=10, maximum=30, step=1, label="Sampling steps | 采样次数", value=20)
                    with gr.Row():
                        cfg = gr.Slider(minimum=0, maximum=20, step=0.1, label="CFG Scale | CFG权重", value=2.5)
                        seed = gr.Slider(minimum=-1, maximum=18446744073709550000, step=1, label="Seed | 种子数", value=-1)
                with gr.Tab(label="输出设置"):
                    with gr.Row():
                        method = gr.Dropdown(["default", "fastest", "slowest"], label="Method | 输出方法", value="default")
                        lossless = gr.Dropdown(["true", "false"], label="Lossless | 无损压缩", value="false")
                    with gr.Row():
                        quality = gr.Slider(minimum=70, maximum=100, step=1, label="Quality | 输出质量", value=85)
                        fps2 = gr.Slider(minimum=1, maximum=30, step=1, label="FPS | 帧率", value=10)
            with gr.Column():
                with gr.Row():
                    btn = gr.Button("Generate | 生成", elem_id="button")
                    btn2 = gr.Button("Interrupt | 终止")
                output = gr.Image(height=600)
        btn.click(fn=SVD.generate, inputs=[SVD.input_image, width, height, video_frames, motion_bucket_id, fps, augmentation_level, min_cfg, seed, steps, cfg, sampler_name, scheduler, denoise, fps2, lossless, quality, method], outputs=[output])
        btn2.click(fn=Function.post_interrupt, inputs=None, outputs=None)


class Extras:
    def generate(initialized, input_image, progress=gr.Progress()):
        module = "Extras"
        if input_image is None:
            return
        else:
            input_image = Function.upload_image(input_image)
        workflow = {}
        node_id = 1
        workflow[str(node_id)] = {"inputs": {"image": input_image, "upload": "image"}, "class_type": "LoadImage"}
        image_port = [str(node_id), 0]
        if initialized is True:
            if module not in Upscale.cache and module not in UpscaleWithModel.cache:
                return
            if module in Upscale.cache:
                workflow, node_id, image_port = Upscale.add_node(module, workflow, node_id, image_port)
            if module in UpscaleWithModel.cache:
                workflow, node_id, image_port = UpscaleWithModel.add_node(module, workflow, node_id, image_port)
        else:
            return
        node_id += 1
        workflow[str(node_id)] = {"inputs": {"filename_prefix": "ComfyUI", "images": image_port}, "class_type": "SaveImage"}
        output = Function.gen_image(workflow, 1, 1, progress)[0]
        if output is not None:
            output = output[0]
        return output
 
    def blocks():
        with gr.Row():
            with gr.Column():
                Extras.input_image = gr.Image(value=None, type="pil")
                with gr.Row():
                    with gr.Tab(label="算术放大"):
                        Upscale.blocks("Extras")
                with gr.Row():
                    with gr.Tab(label="超分放大"):
                        UpscaleWithModel.blocks("Extras")
                gr.HTML("注意：同时启用两种放大模式将先执行算术放大，再执行超分放大，最终放大倍数为二者放大倍数的乘积！")
            with gr.Column():
                with gr.Row():
                    btn = gr.Button("Generate | 生成", elem_id="button")
                    btn2 = gr.Button("Interrupt | 终止")
                output = gr.Image(height=600)
        btn.click(fn=Extras.generate, inputs=[Initial.initialized, Extras.input_image], outputs=[output])
        btn2.click(fn=Function.post_interrupt, inputs=None, outputs=None)


class Info:
    def generate(image_info, progress=gr.Progress()):
        if not image_info or image_info is None or image_info == "仅支持API工作流！！！" or "Version:" in image_info or image_info == "None":
            return
        workflow = json.loads(image_info)
        return Function.gen_image(workflow, 1, 1, progress)[0]
 
    def order_workflow(workflow):
        if workflow is None:
            return gr.update(visible=False, value=None)
        workflow = json.loads(workflow)
        if "last_node_id" in workflow:
            return gr.update(show_label=False, visible=True, value="仅支持API工作流！！！", lines=1)
        workflow = Function.order_workflow(workflow)
        lines = len(workflow) + 5
        workflow_string = "{"
        for node in workflow:
            workflow_string = workflow_string + "\n" + f'"{node}": {workflow[node]},'
        workflow_string = workflow_string + "\n}"
        workflow_string = workflow_string.replace(",\n}", "\n}")
        workflow_string = workflow_string.replace("'", '"')
        return gr.update(label="Ordered workflow_api", show_label=True, visible=True, value=workflow_string, lines=lines)
 
    def get_image_info(image_pil):
        if image_pil is None:
            return gr.update(visible=False, value=None)
        else:
            image_info = Function.get_image_info(image_pil)
            if image_info == "None":
                return gr.update(visible=False, value=None)
            if "Version:" in image_info:
                return gr.update(label="Image info", show_label=True, visible=True, value=image_info, lines=3)
            return Info.order_workflow(image_info)
 
    def hide_another_input(this_input):
        if this_input is None:
            return gr.update(visible=True)
        return gr.update(visible=False)
 
    def blocks():
        with gr.Row():
            with gr.Column():
                Info.input_image = gr.Image(value=None, type="pil")
                workflow = gr.File(label="workflow_api.json", file_types=[".json"], type="binary")
                image_info = gr.Textbox(visible=False)
            with gr.Column():
                with gr.Row():
                    btn = gr.Button("Generate | 生成", elem_id="button")
                    btn2 = gr.Button("Interrupt | 终止")
                output = gr.Gallery(preview=True, height=600)
        btn.click(fn=Info.generate, inputs=[image_info], outputs=[output])
        btn2.click(fn=Function.post_interrupt, inputs=None, outputs=None)
        Info.input_image.change(fn=Info.hide_another_input, inputs=[Info.input_image], outputs=[workflow])
        Info.input_image.change(fn=Info.get_image_info, inputs=[Info.input_image], outputs=[image_info])
        workflow.change(fn=Info.hide_another_input, inputs=[workflow], outputs=[Info.input_image])
        workflow.change(fn=Info.order_workflow, inputs=[workflow], outputs=[image_info])

 
with gr.Blocks(css="#button {background: #FFE1C0; color: #FF453A} .block.padded:not(.gradio-accordion) {padding: 0 !important;} div.form {border-width: 0; box-shadow: none; background: white; gap: 1.15em;}") as demo:
    Initial.initialized = gr.Checkbox(value=False, visible=False)
    with gr.Tab(label="Stable Diffusion"):
        SD.blocks()
    if SC.enable is True:
        with gr.Tab(label="Stable Cascade"):
            SC.blocks()
    if SVD.enable is True:
        with gr.Tab(label="Stable Video Diffusion"):
            SVD.blocks()
    with gr.Tab(label="Extras"):
        Extras.blocks()
    with gr.Tab(label="Info"):
        Info.blocks()
    
    SD.send_to_sd.click(fn=Function.send_to, inputs=[SD.data, SD.index], outputs=[SD.input_image])
    if SC.enable is True:
        SD.send_to_sc.click(fn=Function.send_to, inputs=[SD.data, SD.index], outputs=[SC.input_image])
    if SVD.enable is True:
        SD.send_to_svd.click(fn=Function.send_to, inputs=[SD.data, SD.index], outputs=[SVD.input_image])
    SD.send_to_extras.click(fn=Function.send_to, inputs=[SD.data, SD.index], outputs=[Extras.input_image])
    SD.send_to_info.click(fn=Function.send_to, inputs=[SD.data, SD.index], outputs=[Info.input_image])
    if SC.enable is True:
        SC.send_to_sd.click(fn=Function.send_to, inputs=[SC.data, SC.index], outputs=[SD.input_image])
        SC.send_to_sc.click(fn=Function.send_to, inputs=[SC.data, SC.index], outputs=[SC.input_image])
        if SVD.enable is True:
            SC.send_to_svd.click(fn=Function.send_to, inputs=[SC.data, SC.index], outputs=[SVD.input_image])
        SC.send_to_extras.click(fn=Function.send_to, inputs=[SC.data, SC.index], outputs=[Extras.input_image])
        SC.send_to_info.click(fn=Function.send_to, inputs=[SC.data, SC.index], outputs=[Info.input_image])
 
demo.queue().launch()