# ComfyUI-GTSuya-Nodes

ComfyUI-GTSuya-Nodes is a ComyUI extension designed to add several wildcards supports into [ComfyUI](https://github.com/comfyanonymous/ComfyUI). Wildcards allow you to use \_\_name__ syntax in your prompt to get a random line from a file named name.txt in a wildcards directory.

# Installation
To install ComfyUI-GTSuya-Nodes in addition to an existing installation of ComfyUI, you can follow the following steps:

1. cd custom_nodes
1. git clone https://github.com/GTSuya-Studio/ComfyUI-Gtsuya-Nodes.git
1. Restart ComfyUI

# How to Use
## Wildcards / Simple wildcards
Create a directory named **wildcards** into the Comfyui root folder and put all your wildcards text files into it. Add a **Simple wildcards** node: Right-click > Add Node > GtsuyaStudio > Wildcards > Simple wildcards.

![Simple wildcards](https://github.com/GTSuya-Studio/ComfyUI-Gtsuya-Nodes/assets/29682182/6f319087-3efb-4f63-8489-216909e64085)

Enter your prompt into the text box. Wildcard words must be indicated with double underscore around them. For example, if your wildcards file is named **country.txt**, the corresponding wildcard word must be **\_\_country__**. You can add as many wildcard as you want.

## Wildcards / Simple wildcards (Dir.)
This node is pretty similar to **Simple wildcards**. The only difference is that you can choose any directory on your hard drive containing your wildcards files. Those wildcards files don't need to be specificaly into the Comfyui root directory. To add **Simple wildcards (Dir.)** node: Right-click > Add Node > GtsuyaStudio > Wildcards > Simple wildcards (Dir.).

![Simple wildcards (Dir.)](https://github.com/GTSuya-Studio/ComfyUI-Gtsuya-Nodes/assets/29682182/e9bb74e7-4496-4bba-8477-44dad4639f58)

Enter your prompt into the text box. Wildcard words must be indicated with double underscore around them. For example, if your wildcards file is named **country.txt**, the corresponding wildcard word must be **\_\_country__**. You can add as many wildcard as you want. Then indicate the path of the directory where are stored your wildcards files.

## Wildcards / Wildcards Nodes
This node allows to manage wildcards lists and values directly from ComfyUI workflows. It is not dependent anymore of external text files. This node must work in conjunction with **Ramdom Line** nodes from [Hakkun-ComfyUI-nodes](https://github.com/tudal/Hakkun-ComfyUI-nodes) which return a ramdom string from a list. To add **Wildcards Nodes** node: Right-click > Add Node > GtsuyaStudio > Wildcards > Wildcards Nodes.

![Wildcards](https://github.com/GTSuya-Studio/ComfyUI-Gtsuya-Nodes/assets/29682182/3ff04623-c4ed-470a-b923-469c1d899991)

Enter your prompt into the text box. Wildcard entries (srt#) must be indicated with double underscore around them and must correspond to the **Ramdom Line** nodes linked to this entry. For example, if a **Ramdom Line** nodes is linked to **str3**, the corresponding wildcard word must be **\_\_str3__**. You can actually use a maxiumum of 5 wildcards as the same time.

## Dowmloads / Danbooru (ID)
This node allows to automaticaly get image url and tags list from a post hosted on [Danbooru](https://danbooru.donmai.us/) website. This node could work in conjunction with **Load Image From URL** node from [comfyui-art-venture](https://github.com/sipherxyz/comfyui-art-venture) nodes to import the corresponding image directly into ComfyUI. The tags list could be directly used as a prompt, or part of a prompt. To add **Danbooru (ID)** node: Right-click > Add Node > GtsuyaStudio > Downloads > Danbooru (ID).

![Capture d’écran 2023-11-26 165820](https://github.com/GTSuya-Studio/ComfyUI-Gtsuya-Nodes/assets/29682182/d09a0d22-7d87-4d03-8302-4d305f149f12)

The node can be used by indicating the ID number of the [Danbooru post](https://danbooru.donmai.us/posts) website. 

## Dowmloads / Danbooru (Random)
This node allows to automaticaly get image url and tags list from a random post hosted on [Danbooru](https://danbooru.donmai.us/) website. It is also possible to restrict obtained results to a specific tag query. This node could work in conjunction with **Load Image From URL** node from [comfyui-art-venture](https://github.com/sipherxyz/comfyui-art-venture) nodes to import the corresponding image directly into ComfyUI. The tags list could be directly used as a prompt, or part of a prompt. To add **Danbooru (Random)** node: Right-click > Add Node > GtsuyaStudio > Downloads > Danbooru (Random).

![Capture d’écran 2023-11-26 163744](https://github.com/GTSuya-Studio/ComfyUI-Gtsuya-Nodes/assets/29682182/d6d8fc8a-da16-4403-ab56-1343d00a56e5)

To use this node, you need a valid API key from [Danbooru](https://danbooru.donmai.us/) website. To obtain such API key, you need first to have an [Danbooru account](https://danbooru.donmai.us/users/new), then ask for an unique API key, and then give permission to use **posts:random** data. Once it is done, finaly indicate your Danbooru login and API key number. The node is ready. The node can be used directly with the default settings. If you want to restrict results and obtain a tags list containing a specific tag, indicate this tag into the **tag_query** field. This tag must be a valid [Danbooru tag](https://danbooru.donmai.us/tags) website.

## Tools / Replace Strings
This node allows to automaticaly delete or replace some specific strings into a test or a prompt. To add **Replace Strings** node: Right-click > Add Node > GtsuyaStudio > Tools > Replace Strings.

![Capture d’écran 2023-12-01 202225](https://github.com/GTSuya-Studio/ComfyUI-Gtsuya-Nodes/assets/29682182/dc9409bf-1b12-4571-9deb-755e31e16281)

To use this node, you need to link inputs to 2 text nodes : **text** entry correspond to the text where you want to replace or delete strings, and **replace_list** entry correspond to the list of words that would be replaced. You have to set one strings replacement per line. String replacement line must be like this: **string1|string2**, where **string1** is the  string that will be replaced, and **string2** is the replacement string. If **string2** is not specified, **string1** will be deleted form the text.
 
## Tools / Random File From Path
This is a simple node that return a random file path form a directory. In case of images, this node could work in conjunction with **Load Image From URL** node from [comfyui-art-venture](https://github.com/sipherxyz/comfyui-art-venture) nodes to import the corresponding image directly into ComfyUI. To add **Random File From Path** node: Right-click > Add Node > GtsuyaStudio > Tools > Random File From Path.

![Capture d’écran 2023-12-09 201515](https://github.com/GTSuya-Studio/ComfyUI-Gtsuya-Nodes/assets/29682182/480820a9-5a24-4a90-9eda-0de4e2143561)

To use this node you just have to indicate a directory path where are located the files you want to randomly select. 

