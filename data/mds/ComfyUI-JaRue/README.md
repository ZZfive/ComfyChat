# ComfyUI-JaRue
Takes a YouTube Video ID and pulls a transcript (if available) from the YouTube server and delivers it in a format you can link to AnimatedDiff or any other ComfyUI batch prompt animation. 

![youtube2prompt](https://github.com/jtrue/ComfyUI-JaRue/assets/5502214/d56b845b-2167-4bc7-8630-526810c2190b)

**video_id** - Find this on any YouTube video by visiting the page and extracting it from the url. If a YouTube video does not have a transcript the tool will give you an error telling you. 

**fps** - Frames per second allows you to reduce a higher-framed YouTube transcript down to 10fps which is more friendly for batch animations in ComfyUI. You will see the frame counts return based on your fps setting.

**begin** - If you want to start reading the transcript later in the timeline you can enter a starting frame here and the frame counts will adjust. If you do this more than once you need to keep track of the last number and add it to the new number because the begin frame will always start at the beginning of the YouTube video and not at your last begin point. 

**cache** - Enable cache to avoid spamming YouTube. When you disable cache and press queue the prior cache will be deleted and will force the tool to ask YouTube for the transcripts again. 


![YouTube2Transcript](https://github.com/jtrue/ComfyUI-JaRue/assets/5502214/0085d1bb-7f84-4155-b1b2-85f9f3fe51ee)


You will need the YouTube Transcript API Library installed. See link.  

https://pypi.org/project/youtube-transcript-api/

After following the install instructions you may discover an error in ComfyUI telling you the library is still not found. To fix this issue you will need to manually copy the api library you installed from your python library into your ComfyUI custom_nodes directory. 

There is a Text2Image tool included with this library so you can superimpose the captions on top of the video but i am still working on the code to tie it all together. Right now, you can generate an image with the caption and manually connect that into a batch video. 

![t2icui](https://github.com/jtrue/ComfyUI-JaRue/assets/5502214/4587e0d7-0773-4537-ac20-8ca31ad09170)

If you have trouble with the library installs you can remove the jru_text2image.py file from the __init__.py by commenting out the line and rebooting ComfyUI so you can use the YouTube tool.  

I cover this node here in a video: https://youtu.be/Si4mwBQuzYQ?si=X306LyKT5NgDvP7a
