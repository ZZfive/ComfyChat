import json
from datasets import load_dataset
from utils import load4json

path = '/root/code/ComfyChat/data/message_jsons/v2/comfyui_data_v2_2.json'
# temp_path = "temp.json"

# with open(path, "r") as f:
#     data = json.load(f)

# with open(temp_path, "w") as f:
#     json.dump(data[8343:8350], f)
# dataset = load_dataset("json", data_files=temp_path)
# print("\n-------This works when the JSON file length is 57225-------\n")

# with open(temp_path, "w") as f:
#     json.dump(data[8350:8357], f)
# dataset = load_dataset("json", data_files=temp_path)
# print("\n-------This works and eliminates data issues-------\n")



# for i, v in enumerate(data):
#     if isinstance(v, dict) and "messages" in v and len(v["messages"]) == 2:
#         d1 = v["messages"][0]
#         d2 = v["messages"][1]
#         if ("role" in d1 and d1["role"] and "content" in d1 and d1["content"] and isinstance(d1["content"], str)) and ("role" in d2 and d2["role"] and "content" in d2 and d2["content"] and isinstance(d2["content"], str)):
#             continue
#         else:
#             print(i, v)
#     else:
#         print(i, v)


def is_valid_structure(item: dict) -> bool:
    """
    检查字典是否具有与示例结构一致的格式。
    
    参数:
    item (dict): 需要验证的字典。
    
    返回:
    bool: 如果字典符合结构要求，返回 True，否则返回 False。
    """
    # 检查是否有 "messages" 键，且对应的值是一个列表
    if not isinstance(item, dict) or "messages" not in item or not isinstance(item["messages"], list):
        return False
    
    # 检查 "messages" 列表中的每个元素
    for message in item["messages"]:
        # 每个 message 应该是字典
        if not isinstance(message, dict):
            return False
        # 检查 "role" 和 "content" 键是否存在，且 "role" 是字符串，"content" 也是字符串
        if "role" not in message or "content" not in message:
            return False
        # 检查每个消息是否只包含 "role" 和 "content" 两个键
        if set(message.keys()) != {"role", "content"}:
            return False
        if not isinstance(message["role"], str) or not isinstance(message["content"], str):
            return False
    
    return True


# data = load4json(path)
# # print(data[8350])
# for i, item in enumerate(data):
#     if not is_valid_structure(item):
#         print(item)
#         print('*' * 40)

dataset = load_dataset("json", data_files=path)