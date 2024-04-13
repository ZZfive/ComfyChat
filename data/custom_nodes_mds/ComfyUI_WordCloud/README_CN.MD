# ComfyUI Word Cloud£º
ComfyUIÉú³É´ÊÔÆÍ¼ÏñµÄ²å¼þ£¬ÊÇ [word_cloud](https://github.com/amueller/word_cloud) ÔÚComfyUIÖÐµÄÊµÏÖ¡£
![image](image/comfy_wordcloud_advance.png)

## ¸üÐÂ£º
* Word Cloud½ÚµãÔö¼ÓmaskÊä³ö¡£
![image](image/word_cloud_mask.png)

* Ôö¼ÓRGB Color Picker½Úµã£¬Ñ¡ÔñÑÕÉ«¸ü¼Ó·½±ã¡£

* Í¨¹ý±à¼­²å¼þ¸ùÄ¿Â¼ÏÂµÄfont_dir.ini£¬ÓÃ»§¿ÉÒÔ×Ô¶¨Òå×ÖÌåÄ¿Â¼¡£Ã¿´ÎÆô¶¯comfyUIÊ±£¬Õâ¸öÄ¿Â¼ÏÂ*.ttfºÍ*.otfÎÄ¼þ½«ÊÕ¼¯ÏÔÊ¾ÔÚ²å¼þfont_pathÑ¡ÔñÏîÖÐ¡£
font_dir.iniÄ¬ÈÏÊÇwindowsÏµÍ³×ÖÌåÄ¿Â¼(C:\Windows\fonts)¡£
Èç¹û×Ô¶¨ÒåµÄÄ¿Â¼ÎÞÐ§£¬½«ÆôÓÃ×Ô´øfontÄ¿Â¼£¬Õâ¸öÄ¿Â¼°üº¬Ò»¸öAlibaba-PuHuiTi-Heavy.ttfÎÄ¼þ£¬°æÈ¨¹éÊô°¢Àï°Í°Í£¨ÖÐ¹ú£©ÓÐÏÞ¹«Ë¾£¬ÔÊÐíÈÎºÎ¸öÈËºÍÆóÒµÃâ·ÑÊ¹ÓÃ¡£



### ÖØÒªÌáÐÑ£º¼ÓÔØÖ®Ç°¹¤×÷Á÷±£´æµÄ¾É°æ½Úµã£¬ÐèÒªÖØÐÂÉèÖÃ×ÖÌå¡£
* Ê×ÏÈÉèÖÃºÃfont_dir.ini£¬Æô¶¯comfyUI²¢¼ÓÔØ¹¤×÷Á÷£¬ÔÚWordCloud½ÚµãµÄfont_pathÑ¡ÏîÀ¸ÖÐÖØÐÂÑ¡Ôñ×ÖÌå£¬¼´¿É±ÜÃâ±¨´í¡£

* ±¾´Î¸üÐÂ»ùÓÚ[ZHO-ZHO-ZHO](https://github.com/ZHO-ZHO-ZHO/ComfyUI-Text_Image-Composite)µÄ½¨ÒéºÍ°ïÖú¡£


## ½ÚµãËµÃ÷£º

### Word Cloud:
![image](image/word_cloud.png)
¸ù¾ÝÎÄ±¾ÄÚÈÝÉú³É´ÊÔÆÍ¼Ïñ£¬´ÊµÄ´óÐ¡Óë´ÊÆµÏà¹Ø£¬ÆµÂÊÔ½¸ßÎÄ×ÖÔ½´ó¡£¿É¶¨ÒåÅäÉ«£¬Éè¶¨ÖØµã´Ê£¬Éè¶¨ÅÅ³ý´ÊµÈ¡£Ö§³ÖÊäÈë´øalphaÍ¨µÀµÄÍ¼ÏñÉú³É¾ßÓÐÂÖÀªÐÎ×´µÄ´ÊÔÆÍ¼Ïñ¡£

Ñ¡ÏîËµÃ÷£º   
* color_ref_image: ÊäÈëµÄÍ¼Ïñ×÷ÎªÎÄ×ÖÑÕÉ«µÄ²Î¿¼¡£
* mask_image: ÊäÈëµÄÍ¼Ïñ×÷ÎªÎÄ×ÖÐÎ×´µÄÂÖÀª¡£Èç¹û´øalphaÍ¨µÀ£¬Ê¹ÓÃÍ¨µÀ×÷ÎªÂÖÀª£»²»´øÍ¨µÀµÄ°´ÑÕÉ«¾ö¶¨ÂÖÀªµÄÐÎ×´£¬°×É«µÄ²¿·Ö½«±»ÅÅ³ý¡£
* text: ´Ë´¦µÄÎÄ±¾ÄÚÈÝ½«·Ö½âÎªÖð¸öµ¥´Ê£¬ÕâÐ©µ¥´Ê×÷Îª´ÊÔÆµÄÔªËØ¡£µ¥´Ê³öÏÖÆµÂÊÔ½¸ß£¬ÔÚ´ÊÔÆÉÏÃè»æµÃÔ½´ó¡£
* width: Éú³ÉÍ¼ÏñµÄ¿í¶È¡£(Èç¹ûÓÐmask_imageÊäÈë£¬ÔòÊ¹ÓÃmask_imageµÄ³ß´ç£¬´ËÉèÖÃÊýÖµ±»ºöÂÔ)
* height: Éú³ÉÍ¼ÏñµÄ¸ß¶È¡£(Èç¹ûÓÐmask_imageÊäÈë£¬ÔòÊ¹ÓÃmask_imageµÄ³ß´ç£¬´ËÉèÖÃÊýÖµ±»ºöÂÔ)
* scale: ·Å´óÏµÊý£¬×îÖÕÉú³ÉµÄÍ¼Ïñ´óÐ¡½«ÊÇwidthºÍheightÉè¶¨ÊýÖµÓë´ËÊýÏà³Ë¡£
* margin: ¿Õ°×±ß´óÐ¡¡£
* font_path: ×ÖÌåÎÄ¼þ¡£
* min_font_size: µ¥´ÊÔªËØÏÔÊ¾µÄ×îÐ¡Öµ¡£
* max_font_size£ºµ¥´ÊÔªËØÏÔÊ¾µÄ×î´óÖµ¡£
* relative_scaling: ´ÊÔÆÖÐµ¥´ÊÔªËØµÄÏà¶Ô´óÐ¡¡£ÖµÔ½´ó£¬ÀëÉ¢¶ÈÔ½¸ß¡£
* colormap: ÎÄ×ÖÑÕÉ«¡£ÕâÀïÊ¹ÓÃµÄÊÇMatplotlib Ìá¹©µÄ¶àÖÖÔ¤¶¨Òåcolormap¡£(Èç¹ûÓÐcolor_ref_imageÊäÈë£¬Ôò´ËÉèÖÃ±»ºöÂÔ)
* background_color: ±³¾°ÑÕÉ«£¬Ê¹ÓÃ16½øÖÆRGB¸ñÊ½ÃèÊö¡£(Èç¹ûtransparent_backgroundÉè¶¨ÎªTrue£¬Ôò´ËÉèÖÃ±»ºöÂÔ)
* transparent_background: ±³¾°ÊÇ·ñÍ¸Ã÷¡£´Ë´¦ÉèÎªTrue½«Êä³ö´øalphaÍ¨µÀµÄÍ¼Ïñ¡£
* prefer_horizontal: µ¥´ÊË®Æ½³öÏÖÂÊ¡£×îµÍ0(È«ÊúÅÅ)£¬×î¸ß1(È«ºáÅÅ)¡£
* max_words: µ¥´Ê×î´óÊýÁ¿¡£
* repeat: µ±µ¥´ÊÃ»ÓÐ´ïµ½×î´óÊýÁ¿Ê±ÊÇ·ñÔÊÐíÖØ¸´¡£
* include_numbers: µ¥´ÊÔªËØÊÇ·ñ°üº¬Êý×Ö¡£
* random_state£º¿ØÖÆ´ÊÔÆÉú³É¹ý³ÌÖÐËæ»úÊýÉú³ÉÆ÷µÄÖÖ×Ó¡£ÉèÎª-1ÎªÃ¿´ÎËæ»ú£¬ÆäËûÖµÔòÃ¿´Î¹Ì¶¨¡£
* stopwords: ÆÁ±Î´Ê¡£Éè¶¨ÔÚ´Ë´¦µÄµ¥´Ê²»»áÔÚÍ¼ÖÐ³öÏÖ¡£Ã¿¸ö´ÊÖ®¼äÓÃ¶ººÅ(ÖÐÓ¢ÎÄ¾ù¿É)»ò¿Õ¸ñ·Ö¿ª¡£
* contour_width: ÂÖÀª¿í¶È¡£ÓÐmask_imageÊäÈë²ÅÓÐÐ§¡£
* contour_color: ÂÖÀªÑÕÉ«¡£Ê¹ÓÃ16½øÖÆRGB¸ñÊ½ÃèÊö¡£ÓÐmask_imageÊäÈë²ÅÓÐÐ§¡£
* keynote_words: ÖØµã´Ê¡£ÕâÀïÉè¶¨µÄµ¥´Ê½«¶îÍâ·Å´ó£¬ÔÚstopwordsÖÐÉè¶¨ÁËÍ¬Ñùµ¥´ÊµÄ³ýÍâ¡£Ã¿¸ö´ÊÖ®¼äÓÃ¶ººÅ(ÖÐÓ¢ÎÄ¾ù¿É)»ò¿Õ¸ñ·Ö¿ª¡£
* keynote_weight: ÖØµã´Ê¼ÓÈ¨¡£ÊýÖµÔ½´ó£¬ÖØµã´ÊÏà¶ÔÔ½´ó¡£

Êä³ö£º
* image(Ö§³ÖalphaÍ¨µÀ)
* mask

### RGB Color Picker
![image](image/rgb_color_picker.png)
ÐÞ¸Ä×Ô[mtb nodes](https://github.com/melMass/comfy_mtb)µÄWeb Extensions¡£ÔÚÉ«°åÉÏÑ¡È¡ÑÕÉ«£¬²¢Êä³öRGBÖµ¡£

Ñ¡ÏîËµÃ÷£º   
* mode£º Êä³ö¸ñÊ½£¬ÓÐÊ®Áù½øÖÆ(HEX)ºÍÊ®½øÖÆ(DEC)¿ÉÑ¡Ôñ¡£  

Êä³ö£º 
* string×Ö·û´®¡£

### Load Text File£º
![image](image/load_text_file.png)
´ÓÖ¸¶¨Â·¾¶¼ÓÔØÎÄ±¾ÎÄ¼þ£¬Éè¶¨Ê¹ÓÃutf-8±àÂë¡£

Ñ¡ÏîËµÃ÷£º   
* path£º txtÎÄ¼þÂ·¾¶¡£   

Êä³ö£º 
* string×Ö·û´®¡£

## Ê¹ÓÃÊ¾Àý£º

![image](image/comfy_wordcloud_simple.png)
ÔÚworkflowÄ¿Â¼ÏÂÓÐjson¸ñÊ½µÄ¹¤×÷Á÷Ê¾ÀýÎÄ¼þ¡£

## °²×°·½·¨£º  

* ÔÚCompyUI²å¼þÄ¿Â¼(ÀýÈç¡°CompyUI\custom_nodes\¡±)ÖÐ´ò¿ªcmd´°¿Ú£¬¼üÈë```git clone https://github.com/chflame163/ComfyUI_WordCloud.git```°²×°¡£»òÕßÏÂÔØ½âÑ¹zipÎÄ¼þ£¬½«µÃµ½µÄÎÄ¼þ¼Ð¸´ÖÆµ½ ComfyUI\custom_nodes\
* °²×°ÒÀÀµ°ü£¬ÔÚ×ÊÔ´¹ÜÀíÆ÷ComfyUI\custom_nodes\ComfyUI_WordCloud ²å¼þÄ¿Â¼Î»ÖÃ´ò¿ªcmd´°¿Ú£¬ÊäÈëÒÔÏÂÃüÁî£º  
```..\..\..\python_embeded\python.exe -m pip install -r requirements.txt```
* ÖØÐÂ´ò¿ªComfyUI¡£
