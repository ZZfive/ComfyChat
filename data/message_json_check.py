import json
from datasets import load_dataset

path = '/root/json_check/comfyui_data_v1.json'
temp_path = "temp.json"

with open(path, "r") as f:
    data = json.load(f)

with open(temp_path, "w") as f:
    json.dump(data[:13352], f)
dataset = load_dataset("json", data_files=temp_path)
print("\n-------This works when the JSON file length is 113352-------\n")

with open(temp_path, "w") as f:
    json.dump(data[13352:], f)
dataset = load_dataset("json", data_files=temp_path)
print("\n-------This works and eliminates data issues-------\n")



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