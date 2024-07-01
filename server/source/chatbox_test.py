import random
import gradio as gr

'''
方案一：chatbot中是可以填入音频和视频的，设置一个开关，打开后每次就直接输出音频，但是要注意history要保存文本，需要进一步判断
'''
# def load():
#     return [
#         ("Here's an audio", gr.Audio("https://github.com/gradio-app/gradio/raw/main/test/test_files/audio_sample.wav")),
#         ("Here's an video", gr.Video("https://github.com/gradio-app/gradio/raw/main/demo/video_component/files/world.mp4"))
#     ]

# with gr.Blocks() as demo:
#     chatbot = gr.Chatbot()
#     button = gr.Button("Load audio and video")
#     button.click(load, None, chatbot)

# demo.launch(server_name='0.0.0.0')

'''方案二：chatbot.select事件能监听用户对chatbot中记录的点击动作，点击后生成一个音频，然后返回到一个gr.Audio组件中'''
# 假设你的生成回答的函数
def generate_response(message):
    response = "这是生成的回答"  # 你的算法生成的回答
    return response

# 定义处理选择事件的回调函数
def handle_select(evt: gr.SelectData):
    selected_index = evt.index  # 获取用户选择的对话条目索引
    selected_text = evt.value  # 获取用户选择的对话条目文本
    return f"你选择了第 {selected_index} 条对话: {selected_text}"


with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
    msg_out = gr.Textbox()

    def respond(message, chat_history):
        bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
        chat_history.append((message, bot_message))
        # time.sleep(2)
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])


    # 添加 Chatbot.select 事件监听器
    chatbot.select(handle_select, outputs=msg_out)

# 启动接口
if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0')