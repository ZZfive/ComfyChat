## ¸üÐÂ£º
* MicrosoftSpeech_TTS½ÚµãÉÏµÄÓïÒôÁÐ±í¿ÉÒÔÍ¨¹ýÐÞ¸Ävoicelist.ini×Ô¶¨Òå¡£Õâ¸öÎÄ¼þÎ»ÓÚ²å¼þµÄ¸ùÄ¿Â¼£¬°üº¬È«²¿¿ÉÓÃÓïÒôÑ¡Ïî²¢ÇÒÄ¬ÈÏÆÁ±ÎÁËÒ»²¿·ÖÄÚÈÝ¡£ÓÃ¼ÇÊÂ±¾»òÆäËûÎÄ±¾±à¼­Èí¼þ´ò¿ªÕâ¸öÎÄ¼þ£¬ÈôÏëÆôÓÃÄ³¸öÓïÒô£¬É¾³ýÕâÒ»ÐÐ¿ªÍ·µÄ¡°#¡±¡£¶ÔÆôÓÃµÄÓïÒô½øÐÐÆÁ±ÎÔòÔÚÕâÐÐ¿ªÍ·Ìí¼Ó¡°#¡±¡£
* ÓïÒôÁÐ±í¸ÄÎª¸üÇåÎúµÄÃû³ÆÏÔÊ¾£¨ÓïÖÖÓïÑÔ+ÐÕÃû+ÐÔ±ð£©

## Ôö¼Ó½Úµã£º
ÓÉÓÚÒýÈëÁËÐÂµÄ½Úµã£¬ÇëÖØÐÂ°²×°ÒÀÀµ°ü¡£
### Play Sound 
¿É´¥·¢µÄÉùÒô²¥·Å½Úµã£¬Ö§³Ömp3ºÍwav¸ñÊ½¡£Õâ¸ö½ÚµãÖ§³Ö¶àÏß³Ì²¥·Å¡£  
Ñ¡ÏîËµÃ÷£º  
* path£ºÉùÒôÎÄ¼þÂ·¾¶¡£  
* volume£ºÒôÁ¿µ÷Õû·¶Î§0-1.0¡£  
* speed£ºÓïËÙµ÷Õû·¶Î§0.1-2.0¡£  
* trigger£º´¥·¢¿ª¹Ø£¬µ±ÆäÖµÎªTrueÊ±¿ªÊ¼²¥·Å¡£
### Play Sound(loop)
¿É´¥·¢µÄÉùÒô²¥·Å½Úµã£¬Ö§³Ömp3ºÍwav¸ñÊ½¡£Õâ¸ö½ÚµãÊ¼ÖÕÕ¼ÓÃÒ»¸öÉùÒô²¥·ÅÏß³Ì¡£  
Ñ¡ÏîËµÃ÷£º  
* path£ºÉùÒôÎÄ¼þÂ·¾¶¡£  
* volume£ºÒôÁ¿µ÷Õû·¶Î§0-1.0¡£  
* loop£ºµ±ÆäÖµÎªTrueÊ±Ñ­»·²¥·Å£¬·ñÔò²¥·ÅÒ»´Î¡£  
* trigger£º´¥·¢¿ª¹Ø£¬µ±ÆäÖµÎªTrueÊ±¿ªÊ¼²¥·Å¡£

### Input Trigger
ÊäÈë´¥·¢Æ÷£¬¿É½ÓÈëÈÎÒâÀàÐÍµÄÊý¾Ý£¬µ±¼ì²âµ½ÓÐÊäÈëÄÚÈÝ£¨·ÇNone£©Ê±Êä³öTrue£»Èç¹ûÃ»ÓÐ½ÓÈëÊäÈë£¬½«Ò»Ö±Êä³öFalse¡£  
ÊäÈë£ºÈÎÒâÀàÐÍ£¬°üÀ¨ÇÒ²»ÏÞÓÚimage, latent, model, clip, string, float, intµÈµÈ¡£
Êä³ö£ºBooleanÖµ¡£  
Ñ¡ÏîËµÃ÷£º  
* always_true£ºµ±´ËÑ¡Ïî´ò¿ªÊ±£¬½«ºöÂÔÊäÈë¼ì²â£¬Ö±½ÓÊä³öTrueÖµ£»


# ComfyUI_MSSpeech_TTS  
ComfyUIÏÂÊ¹ÓÃµÄÎÄ±¾×ªÓïÒô²å¼þ¡£Ê¹ÓÃMicrosoft speech TTS ½Ó¿Ú½«ÎÄ±¾ÄÚÈÝ×ªÎªMP3¸ñÊ½µÄÓïÒôÎÄ¼þ¡£  


Ñ¡ÏîËµÃ÷£º   
* voice£º ÓïÒôÖÖÀà¡£   
* rate£º ÓïÒôËÙ¶È¡£Ä¬ÈÏÊÇ0£¬µ÷Õû·¶Î§´Ó-200µ½200¡£Êý×ÖÔ½´óËÙ¶ÈÔ½¿ì¡£  
* filename_prefix£ºÎÄ¼þÃûÇ°×º¡£   

Êä³ö£º 
MP3 File,×Ö·û´®ÀàÐÍ£¬ÆäÄÚÈÝÊÇÓïÒôÎÄ¼þµØÖ·¡£

## Ê¹ÓÃÊ¾Àý£º

![image](image/triggernode.png)
## °²×°·½·¨£º  
- ½âÑ¹zipÎÄ¼þ£¬½«"ComfyUI_MSSpeech_TTS"ÎÄ¼þ¼Ð¸´ÖÆµ½ ComfyUI\custom_nodes\
- °²×°ÒÀÀµ°ü£¬ÔÚ×ÊÔ´¹ÜÀíÆ÷ComfyUI\custom_nodes\ComfyUI_MSSpeech_TTS\ Õâ¸öÎ»ÖÃ´ò¿ªcmd´°¿Ú£¬ÊäÈëÒÔÏÂÃüÁî£º  
```..\..\..\python_embeded\python.exe -s -m pip install -r requirements.txt```
- ÖØÐÂ´ò¿ªComfyUI
