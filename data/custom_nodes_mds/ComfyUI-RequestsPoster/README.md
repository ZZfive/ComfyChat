# ComfyUI-RequestsPoster
画像の入力に反応して指定のURLにrequests.post(url,{key:value})するだけの機能です。<br>
This custom node is that simply posts HttpRequest from ComfyUI.<br>

# インストール方法
下記のどちらかでインストールできます<br>
A）ComfyUI-Managerの「Install via Git URL」でこのリポジトリのクローン用URLをコピペしてください。<br>
　　クローン用URLはこのページの上部にある緑色のボタン「<> Code」を押すと表示されます。<br>
<br>
B）ComfyUI > Cutom_nodesのフォルダをコマンドプロンプトで開き下記コマンドを実行してください。<br>
git clone https://github.com/aburahamu/ComfyUI-RequestsPoster.git<br>
<br>
これでインストールは完了です。ComfyUIを再起動してください。<br>

# 使い方
1)AddNode > RequestsPoster > PostRequest でノードを追加する<br>
2)anyにトリガーとしたいノードを繋ぐ<br>
3)urlにリクエストを投げたいURLをコピペする　例）ディスコードのウェブフックURL<br>
4)keyとvalueにそれぞれリクエストに含めたい文字列を入力する　例）key = content、value = 画像が出来ました<br>
5)Queueする<br>

# 使用例<br>
DiscordのサーバーにWebhookを使ってメッセージを投稿させられます。<br>

# WebhookURLの取得方法<br>
1)左下の「＋」を押してサーバーを追加して、作られたサーバーの歯車アイコンを押す<br>
![02_addServer](https://github.com/aburahamu/ComfyUI-RequestsPoster/assets/166828042/a9c2b8d9-0a21-4eeb-9409-6c5a82a3b9d4)

2)連携サービス　→　ウェブフック　と押す<br>
![03_addWebhook](https://github.com/aburahamu/ComfyUI-RequestsPoster/assets/166828042/89b17581-51ab-404a-9d94-2117c4ec25d6)

3)ウェブフックを開き「URLをコピー」をクリック<br>
![04_copyURL](https://github.com/aburahamu/ComfyUI-RequestsPoster/assets/166828042/c8ebcd28-7464-4ebb-ab0c-8d0f7b16a443)

4)AddNode > RequestsPoster > PostRequest　でノードを追加しanyにトリガーとしたいノードを繋ぎ、URLにウェブフックのURLをコピペする<br>
![01_node](https://github.com/aburahamu/ComfyUI-RequestsPoster/assets/166828042/36fb87ad-21a6-49ca-b145-d2e7f583b322)

5)画像を生成すると、ディスコードにメッセージが投稿されます<br>
![05_HelloWorld](https://github.com/aburahamu/ComfyUI-RequestsPoster/assets/166828042/058960e2-0983-4b8c-be35-ca5bd2aa7cb0)
