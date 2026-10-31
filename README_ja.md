<h1 align="center">HairDeck: Animation-Ready Stylized Hair Cards<br>from a Single Image</h1>

<p align="center"><a href="https://hyzcluster.github.io/">Yuze He</a><sup>1,*</sup> &nbsp; Yixiang Wei<sup>1,*</sup> &nbsp; <a href="https://thuzhaowang.github.io/">Wang Zhao</a><sup>1</sup> &nbsp; <a href="https://czzzzh.github.io/">Chuhao Chen</a><sup>2</sup> &nbsp; <a href="https://amandaynzhou.github.io/">Yanning Zhou</a><sup>3,†</sup> &nbsp; <a href="https://yongjinliu.github.io/">Yong-Jin Liu</a><sup>1,†</sup></p>
<p align="center"><sup>1</sup>Tsinghua University &nbsp; <sup>2</sup>Carnegie Mellon University &nbsp; <sup>3</sup>The Chinese University of Hong Kong</p>
<p align="center"><sup>*</sup>同等の貢献。 &nbsp; <sup>†</sup>責任著者。</p>

<p align="center">
  <a href="https://hyzcluster.github.io/HairDeck"><img src="https://img.shields.io/badge/Project-Page-426b92" alt="プロジェクトページ"></a>
  <a href="https://arxiv.org/abs/2610.40599"><img src="https://img.shields.io/badge/arXiv-2610.40599-b31b1b" alt="arXiv: 2610.40599"></a>
  <a href="https://huggingface.co/hyz317/HairDeck"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Models-Hugging%20Face-ffd21e" alt="Hugging Face のモデルの重み"></a>
  <a href="https://huggingface.co/spaces/hyz317/HairDeck"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Demo-Try%20it-ffd21e" alt="オンラインデモを試す"></a>
  <a href="#video"><img src="https://img.shields.io/badge/Video-Watch-426b92" alt="デモ動画を見る"></a>
</p>

<p align="center"><a href="README.md">English</a> · <a href="README_zh-CN.md">简体中文</a> · <strong>日本語</strong></p>

<a name="video"></a>

## 動画

https://github.com/user-attachments/assets/3eec4de5-ed21-4b75-b7d7-4bd599cbfeb2

## 概要

**HairDeck は、1 枚の画像からアニメ調の毛束メッシュと、それを動かすスプリングボーンのチェーンを生成します。**
本論文では、個々の毛束を表すメッシュを hair card（ヘアカード）と呼びます。
各ヘアカードを 1 つのトークンで表し、set diffusion Transformer でカードをまとめて生成します。
生成した髪型は GLB メッシュとして出力できるほか、アバターに取り付けて VRM として出力することもできます。

- **ヘアカードを集合として生成。** Card VAE が各カードをエンコードし、条件付き set diffusion モデルが髪型全体を生成します。
- **UV 座標付きのジオメトリ。** 出力には個々の毛束が保持され、毛束ごとの UV 座標が含まれます。
- **スプリングボーンによるアニメーション。** chain diffusion model が毛束を動かすボーンを生成し、スキニングと VRM 出力に対応します。

<p align="center">
  <img src="https://huggingface.co/hyz317/HairDeck/resolve/main/assets/method.png" width="100%" alt="HairDeck の処理の流れ：ヘアカードのトークン、画像または点群を条件とする set diffusion、スプリングボーンのチェーンを備えた毛束の生成。">
</p>

## はじめに

[🛠️ インストール](#installation) · [🤗 モデルの重み](#model-weights) · [🚀 クイックスタート](#quick-start) · [🖼️ ローカルデモ](#local-demo) · [📦 出力ファイル](#outputs) · [📊 評価](#test-set-evaluation) · [🏋️ 学習](#training) · [📝 引用](#citation)

<a name="installation"></a>

### 🛠️ インストール

Linux x86-64、Python 3.12、NVIDIA CUDA 対応 GPU 1 基を使用してください。このリリースは PyTorch 2.8 と CUDA 12.8 で動作確認済みです。
動画プレビューには、`PATH` の通った場所に `ffmpeg` も必要です。

```bash
git clone https://github.com/hyz317/HairDeck.git
cd HairDeck

python -m pip install -r requirements.txt
python -m pip install --no-deps pyrender==0.1.45  # without its old PyOpenGL pin
```

以下のコマンドは、同じ Python 環境を使い、リポジトリのルートディレクトリから実行してください。

<a name="model-weights"></a>

### 🤗 モデルの重み

チェックポイントと関連アセットは [hyz317/HairDeck](https://huggingface.co/hyz317/HairDeck) で公開しています。
SAM 3 と DINOv3 のダウンロードにはアクセス許可が必要です。[facebook/sam3](https://huggingface.co/facebook/sam3)
と [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) でそれぞれのライセンスに同意してから、ログインしてください。

```bash
hf auth login
export HAIRDECK_ROOT=/path/to/workspace  # checkpoints, data, third-party weights and outputs
python scripts/setup/download_weights.py
```

ダウンロードスクリプトは、HairDeck のチェックポイント、前処理モデルと画像エンコーダの重み、関連データ、
および VRM Add-on を組み込んだ Blender 4.2 LTS をセットアップします。Blender はデフォルトで Linux x86-64 版をダウンロードします。
VRM Add-on を有効にした既存の Blender 4.2 LTS を使う場合は、`HAIRDECK_BLENDER` にその起動コマンドを設定してください。

デフォルトのワークスペース構成は次のとおりです。

```text
$HAIRDECK_ROOT/
├── ckpt/                       # HairDeck のチェックポイントと潜在表現の統計量
├── data/                       # 正規化の統計量、アバター、デモ用サンプル
└── third_party/
    ├── weights/
    │   ├── dinov3/             # facebook/dinov3-vitl16-pretrain-lvd1689m
    │   ├── sam3/               # facebook/sam3
    │   └── birefnet_matting/    # ZhengPeng7/BiRefNet-matting
    ├── dinov3_src/             # DINOv3 のソースコード
    ├── sam3_src/               # SAM 3 のソースコード
    └── blender/                # Blender と VRM Add-on
```

各パスは環境変数または JSON ファイルで変更できます。
[configs/paths.example.json](configs/paths.example.json) と [hairdeck/paths.py](hairdeck/paths.py) を参照してください。

<a name="quick-start"></a>

### 🚀 クイックスタート

キャラクター画像から 4 つのバリエーションを生成し、バリエーション 0 を VRM アバターと頭部の回転プレビュー動画として出力します。

```bash
python scripts/image_to_hair.py character.png --out results/my_character --n 4
python -m hairdeck.vrm.export results/my_character/character 0
```

結果は `OUT/<image name>/` に保存されます。この例では `results/my_character/character/` です。
`--n` では生成するバリエーション数を 1〜4 に指定でき、`--seed` では最初のシードを設定できます。毛束のメッシュのみを生成する場合は `--no-chains` を指定してください。
独自のモデルを使う場合は、`--ckpt` と、同じ Card VAE で学習した対応する `--chain_ckpt` を指定してください。

<a name="local-demo"></a>

### 🖼️ ローカルデモ

```bash
python -m pip install -r requirements-demo.txt
python demo/app.py --port 7860
```

[http://127.0.0.1:7860](http://127.0.0.1:7860) を開くと、髪型の生成、バリエーションの確認、VRM やアニメーションプレビューの出力ができます。
[オンラインデモ](https://huggingface.co/spaces/hyz317/HairDeck) も利用できます。

<a name="outputs"></a>

## 📦 出力ファイル

シード 0 の主な出力ファイルは次のとおりです。

```text
results/my_character/character/
├── input.png                   # 入力画像
├── prep.png                    # 前処理のマスクと切り抜き画像
├── model_input.png             # モデルに入力する切り抜き画像
├── cards_seed0.json             # 毛束のジオメトリ
├── chains_seed0.json            # 生成されたスプリングボーンのチェーン
├── hair_seed0.glb               # head proxy、毛束ごとの UV 座標、髪色を含む髪型
├── hair_seed0_nohead.glb         # head proxy を含まない髪型
├── info.json                    # 生成設定と結果のメタデータ
├── avatar_seed0.vrm             # VRM 出力コマンドで生成
└── turn_seed0.mp4               # 頭部の回転プレビュー動画の出力コマンドで生成
```

ほかのバリエーションでは、対応するシードがファイル名に使われます。`--no-chains` を指定した場合、チェーンのファイルは生成されません。

<a name="test-set-evaluation"></a>

## 📊 テストセットでの評価

評価用アセットをダウンロードし、点群モデルとチェーンモデルのテストを実行します。各テストではシード 0、1、2 でサンプリングし、
論文に対応する評価指標を表示するとともに、サンプルごとの結果とその平均を `eval.json` に保存します。

```bash
python scripts/setup/download_weights.py --only test_data charm_test_pc dora
python scripts/test.py pointcloud
python scripts/test.py chains                                # On the artists' cards
python scripts/test.py chains --cards results/pointcloud     # On the point-cloud model's cards
```

`results/pointcloud` などの評価用パスは、`HAIRDECK_ROOT` 以下のワークスペース構成と、個別に変更したパスの設定に従います。

<a name="training"></a>

## 🏋️ 学習

CHARM データセットとキャラクターの VRM ファイルを用意し、次のコマンドを実行します。

```bash
python scripts/setup/download_weights.py --only dora test_data charm_test_pc
python scripts/data.py DATASET_DIR VRM_DIR
bash scripts/train/launch_vae.sh
bash scripts/train/launch_pc_dit.sh && bash scripts/train/launch_img_dit.sh && bash scripts/train/launch_chain_dit.sh
```

学習スクリプトの出力先は `ckpt/train_*` で、ダウンロードしたチェックポイントとは別に保存されます。現在の画像モデルの学習設定には synthetic appearance augmentation が含まれます。公開した画像モデルの学習に使用した、スタイルを変換したレンダリング画像群は、
このリリースには含まれていません。

<a name="citation"></a>

## 📝 引用

```bibtex
@misc{hairdeck2026,
  title         = {HairDeck: Animation-Ready Stylized Hair Cards from a Single Image},
  author        = {Yuze He and Yixiang Wei and Wang Zhao and Chuhao Chen and Yanning Zhou and Yong-Jin Liu},
  year          = {2026},
  eprint        = {2610.40599},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2610.40599},
  url           = {https://arxiv.org/abs/2610.40599}
}
```

## 謝辞

HairDeck は [CHARM](https://huggingface.co/datasets/hyz317/CHARM)、[Dora-VAE](https://huggingface.co/Seed3D/Dora-VAE-1.1)、[DINOv3](https://github.com/facebookresearch/dinov3)、[SAM 3](https://github.com/facebookresearch/sam3)、[BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet-matting)、[VRM Add-on for Blender](https://github.com/saturday06/VRM-Addon-for-Blender) を利用しています。これらのリソースを公開してくださった開発者・著者の皆様に感謝します。

## ライセンス

HairDeck のコードとチェックポイントは [Apache-2.0](LICENSE) ライセンスで公開しています。
サードパーティのモデル、コード、データセットには、それぞれのライセンスが適用されます。
