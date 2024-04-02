import os
import re
import json

import requests
from urllib.parse import quote, unquote
import whisperx
from pytube import YouTube

base_audio_dir = '/root/code/ComfyChat/data/audio'
base_txt_dir = '/root/code/ComfyChat/data/audio2txt'


def bilibili_audio_download(url: str) -> str:
    session = requests.session()
    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36 Edg/106.0.1370.37',
            "Referer": "https://www.bilibili.com",
            }
    resp = session.get(url,headers=headers)

    title = re.findall(r'<title data-vue-meta="true">(.*?)_哔哩哔哩_bilibili', resp.text)[0]   
    play_info = re.findall(r'<script>window.__playinfo__=(.*?)</script>', resp.text)[0]

    
    json_data = json.loads(play_info) 
    audio_url = json_data['data']['dash']['audio'][0]['backupUrl'][0]  #音频地址  [0]清晰度最高
    audio_content = session.get(audio_url,headers=headers).content  #音频二进制内容
    save_path = os.path.join(base_audio_dir, f"{quote(url, safe='')}.mp3")
    with open(save_path, 'wb') as f:
        f.write(audio_content)

    return save_path


def youtube_audio_download(url: str): # 需要魔法
    # 创建 YouTube 对象
    yt = YouTube(url)
    
    # 选择最高质量的音频流，并指定格式为 MP3
    audio_stream = yt.streams.filter(only_audio=True).first()
    
    # 下载音频流
    filename = f"{quote(url, safe='')}.mp3"
    audio_stream.download(output_path=base_audio_dir, filename=filename)

    # 获取下载后的文件路径
    audio_file_path = os.path.join(base_audio_dir, audio_stream.default_filename)

    return audio_file_path


def whisperx_transcribe(audio_path: str, model_path: str = '/root/code/ComfyChat/data/weights/models--Systran--faster-whisper-medium/snapshots/08e178d48790749d25932bbc082711ddcfdfbc4f',
                 batch_size: int = 16, compute_type: str = 'float16', device = "cuda") -> None:
    # model = whisperx.load_model("medium", device, compute_type=compute_type, download_root='/root/code/ComfyChat/data/weights')  # large-v2
    model = whisperx.load_model(model_path, device, compute_type=compute_type)

    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size)
    
    txts = []
    for res in result['segments']:
        txts.append(res['text'])

    final_txt = '\n'.join(txts)
    filename = os.path.basename(audio_path)
    filename = os.path.splitext(filename)[0].strip()
    with open(os.path.join(base_txt_dir, f"{filename}.txt"), 'w', encoding='utf-8') as f:
        f.write(final_txt)


def audio_to_txt(url: str, is_bili: bool = True) -> None:
    if is_bili:
        print(f"传入{url}为bilibili视频url，使用对应函数下载音频")
        audio_path = bilibili_audio_download(url)
    else:
        print(f"传入{url}为youtube视频url，使用对应函数下载音频")
        audio_path = youtube_audio_download(url)

    print(f"{url}中音频下载成功，使用whisperx转为文本")
    whisperx_transcribe(audio_path)


if __name__ == '__main__':
    bili_url = 'https://www.bilibili.com/video/BV1zS421A7PG/?spm_id_from=333.999.0.0'
    # youtube_url ='https://www.youtube.com/watch?v=haDxwOgmTyY'
    # youtube_audio_download(bili_url)
    # whisperx_transcribe('/root/code/ComfyChat/data/audio/https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DhaDxwOgmTyY.mp3')
    audio_to_txt(bili_url)

    # data = {'segments': [{'text': '先说几条重要消息小日子疑似在做战前准备本周小日子采取了一系列举措来加强战备第一是在日本规划建设16个军用港口和机场第二是取消高端武器出口限制第三是签署购买老美的500枚战俘远程导弹和14套战术战俘武器控制系统的协议以及其他后勤服务等', 'start': 0.009, 'end': 21.442}, {'text': '要知道战斧远程巡航导弹最高射程1600公里这是能打到东方境内的并且美国海军在本周开始对小日子海上自卫队人员进行战斧巡航导弹训练第四扩编两栖部队并在琉球部署远程导弹而朝鲜这边居韩联社报道朝鲜在加快军工厂项目建设朝鲜大概已动员了约4万名士兵组建了总共20个临时军事团体', 'start': 21.442, 'end': 46.476}, {'text': '然后到处建军工厂扩大弹药装备产能据南华早报报道东方大国又出新神器了研制出了一种射程超过2000公里的防空导弹该防空导弹长8米重2.5吨可车载移动发射能够击落预警机和战略轰炸机而且是两年前就研制成功了那现在是不是批量生产了呢现在外网出现一个有趣的消息', 'start': 46.476, 'end': 70.981}, {'text': '一張新的衛星圖像展示了位於東方境內某個地方複製了臺灣省政府重要地區的詳細模型並且是在一個疑似空中轟炸核射擊訓練場裡所以該模型可能是專門為跑訓練而設計的詳細信息就不多說了然後臺灣省對此做出回應表示已進入戰備狀態', 'start': 70.981, 'end': 90.213}, {'text': '同时宣布台瓦省军队头子将于下周去找美帝中东这边巴雷斯坦哈马斯又干了一件漂亮的伏击哈马斯在汉尤尼斯北部一座建筑物里安装了重型炸药然后设置好诱饵吸引了一队义军过去探查然后触发爆炸造成10名义军伤亡在今年1月也发生过类似事件', 'start': 90.213, 'end': 110.077}, {'text': '那次袭击造成21名以军死亡,十几名重伤以色列从约弹空袭击叙利亚的阿雷坡国际机场和奈拉布军用机场以及一个军用武器库造成至少38名平民和6名真主党成员死亡其中包含真主党火箭弹副司令这件事对真主党来说挺严重的我估计后续真主党会大规模报复性打击据每日电讯报报道以色列承认因为美国背弃了它', 'start': 110.077, 'end': 136.305}, {'text': '他可能无法摧毁哈马斯此外大多数战时内阁部长认为内塔尼亚胡当初决定不派遣代表团是个错误插播一条娱乐新闻作为乌蒙海军行动的一部分爱沙尼亚将派遣一名士兵保护红海船只免受胡塞武装的袭击绷不住了司令们派一个士兵啊', 'start': 136.305, 'end': 154.684}, {'text': '每天看到波罗三傻就感觉看娱乐新闻一样快乐而无方面据无方消息来源监听到至少六架俄罗斯-295MS战略轰炸机从科拉半岛鲍莱尼亚机场起飞与此同时数十架无人机前往乌克兰境内然后俄军从黑海水下发射十几枚巡航导弹目前消息是赫梅利尼茨基利涅波罗、文尼查、日托米尔利沃夫和切尔卡瑟均发生多处爆炸目标全是关键基础设施', 'start': 154.684, 'end': 183.183}, {'text': '顺便说一句,有导弹经过利沃福右拐弯回来,吓得波兰再次派战斗机紧急升空。这大猫天天吓波兰,现在乌克兰防空系统快成筛子了,俄军的直升机每天用数千枚弹药沿着整个前线轰炸。据经济学人报道,俄罗斯正在为前线的新一轮大规模攻势做准备。然后,国防部发布消息称,装备有亚尔斯机动陆基弹道导弹系统的兵团已开启备战状态。', 'start': 183.183, 'end': 208.387}, {'text': '據公開來源的信息,亞爾斯導彈射程最遠可達1.2萬公里,能夠攜帶數枚威力達500千噸的核彈頭。此外,俄羅斯薩爾馬特洲際彈道導彈也開始執行戰鬥任務,而且有消息稱這款導彈還將研製飛核彈頭。', 'start': 208.387, 'end': 223.763}, {'text': '大毛公開這些信息主要是威懾作用烏克蘭外交部長德米特里·庫萊巴表示烏克蘭可以在第一次峰會後就和平方案與俄羅斯進行溝通但是對於現在掌握主動優勢的大毛來說想和談美滿因為大毛現在是越打越要而且國內戰時經濟和各派勢力也不會允許其停下', 'start': 223.763, 'end': 243.677}, {'text': '德国对乌克兰新的军事援助又来了包括1.8万枚155毫米口径炮弹和德国豹2A6坦克的弹药以及军用车辆和多种无人机等等截至目前德国已经提供了价值约280亿物源的军事源土然后法国说2024年生产的80%炮弹将运往乌克兰所有新生产的凯撒火炮将转移给乌军但是搞笑的是法国155毫米口径炮弹一天产量也就100枚', 'start': 243.677, 'end': 271.903}, {'text': '你没看错,就是一百枚。这个是甚至被乌克兰嘲讽过,说凯撒火炮产量还不如他们自己手搓2S22块。则连斯基自己也出来发话了,说乌克兰现在的火炮核弹药数量是欧洲的耻辱。哎呀,斯基,你把大实话说出来了。另外,则连斯基的总统期限在5月21日就到期了,并且斯基决定不举行总统选举。', 'start': 272.022, 'end': 295.247}, {'text': '然后俄罗斯外长拉夫·罗夫说5月20日之后讨论泽联司机统治的合法性可能没有必要因为先活到那时候再说', 'start': 295.247, 'end': 303.097}], 'language': 'zh'}
    # for d in data['segments']:
    #     print(d['text'])
    #     print('')

    # url = 'https://www.bilibili.com/video/BV1CD421V7qe?spm_id_from=333.1007.tianma.2-2-5.click'
    # audio_to_txt(url)