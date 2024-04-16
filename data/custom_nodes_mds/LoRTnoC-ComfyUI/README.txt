

LORTNOC-COMFYUI


[Hugging Face Model]

LoRTnoC(LoRA with hint block of
ControlNet)をComfyUI上で使うためのリポジトリです。

モデルファイルはcontrolnetと同じ場所に入れてください（適当すぎ？）



例


画像にワークフローがついています(多分)。

                      reference         generated
  ------------------- ----------------- ---------------------------
  canny               [canny]           [canny_generated]
  depth               [depth]           [depth_generated]
  hed                 [hed]             [hed_generated]
  fake_scribble       [fake_scribble]   [fake_scribble_generated]
  lineart_anime       [lineart_anime]   [lineart_anime_generated]
  pose(cherry pick)   [pose]            [pose_generated]
