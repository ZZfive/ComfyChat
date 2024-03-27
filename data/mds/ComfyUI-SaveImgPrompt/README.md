## ComfyUI-SaveImgPrompt
Save a png or jpeg and option to save prompt/workflow in a text or json file for each image in Comfy + Workflow loading

## Warning: 

I debugged it mostly with simple workflows. Please open a complaint on github if you have suggestions or complaints.

### Known issues:

I dont know of any yet.

## Description:

This adds a custom node to  Save a png or jpeg and option to save prompt/workflow in a text or json file for each image in Comfy + Workflow loading. Also allows to turn off saving prompt as well as previews and choosing which folder to save it to.

I have added this node to the IO category
![image](https://github.com/mpiquero7164/ComfyUI-SaveImgPrompt/assets/28360938/29e9b0cb-6993-4896-80c1-e150fddeaabf)


## Installation: 

Use git clone https://github.com/mpiquero7164/ComfyUI-SaveImgPrompt.git in your ComfyUI custom nodes directory
Restart Comfyui and you're good to go

## Text Tokens
Text tokens can be used.

Built-in Tokens
[time]
The current system microtime
[time(format_code)]
The current system time in human readable format. Utilizing datetime formatting
Example: [hostname]_[time]__[time(%Y-%m-%d__%I-%M%p)] would output: SKYNET-MASTER_1680897261__2023-04-07__07-54PM
[hostname]
The hostname of the system executing ComfyUI
[user]
The user that is executing ComfyUI

Directive	Meaning	Example	Notes
%a	Weekday as locale’s abbreviated name.	Sun, Mon, …, Sat (en_US); So, Mo, …, Sa (de_DE)	(1)
%A	Weekday as locale’s full name.	Sunday, Monday, …, Saturday (en_US); Sonntag, Montag, …, Samstag (de_DE)	(1)
%w	Weekday as a decimal number, where 0 is Sunday and 6 is Saturday.	0, 1, …, 6	
%d	Day of the month as a zero-padded decimal number.	01, 02, …, 31	(9)
%b	Month as locale’s abbreviated name.	Jan, Feb, …, Dec (en_US); Jan, Feb, …, Dez (de_DE)	(1)
%B	Month as locale’s full name.	January, February, …, December (en_US); Januar, Februar, …, Dezember (de_DE)	(1)
%m	Month as a zero-padded decimal number.	01, 02, …, 12	(9)
%y	Year without century as a zero-padded decimal number.	00, 01, …, 99	(9)
%Y	Year with century as a decimal number.	0001, 0002, …, 2013, 2014, …, 9998, 9999	(2)
%H	Hour (24-hour clock) as a zero-padded decimal number.	00, 01, …, 23	(9)
%I	Hour (12-hour clock) as a zero-padded decimal number.	01, 02, …, 12	(9)
%p	Locale’s equivalent of either AM or PM.	AM, PM (en_US); am, pm (de_DE)	(1), (3)
%M	Minute as a zero-padded decimal number.	00, 01, …, 59	(9)
%S	Second as a zero-padded decimal number.	00, 01, …, 59	(4), (9)
%f	Microsecond as a decimal number, zero-padded to 6 digits.	000000, 000001, …, 999999	(5)
%z	UTC offset in the form ±HHMM[SS[.ffffff]] (empty string if the object is naive).	(empty), +0000, -0400, +1030, +063415, -030712.345216	(6)
%Z	Time zone name (empty string if the object is naive).	(empty), UTC, GMT	(6)
%j	Day of the year as a zero-padded decimal number.	001, 002, …, 366	(9)
%U	Week number of the year (Sunday as the first day of the week) as a zero-padded decimal number. All days in a new year preceding the first Sunday are considered to be in week 0.	00, 01, …, 53	(7), (9)
%W	Week number of the year (Monday as the first day of the week) as a zero-padded decimal number. All days in a new year preceding the first Monday are considered to be in week 0.	00, 01, …, 53	(7), (9)
%c	Locale’s appropriate date and time representation.	Tue Aug 16 21:30:00 1988 (en_US); Di 16 Aug 21:30:00 1988 (de_DE)	(1)
%x	Locale’s appropriate date representation.	08/16/88 (None); 08/16/1988 (en_US); 16.08.1988 (de_DE)	(1)
%X	Locale’s appropriate time representation.	21:30:00 (en_US); 21:30:00 (de_DE)	(1)
%%	A literal '%' character.	%	
