<h1 align="center">HairDeck: Animation-Ready Stylized Hair Cards<br>from a Single Image</h1>

<p align="center"><a href="https://hyzcluster.github.io/">Yuze He</a><sup>1,*</sup> &nbsp; Yixiang Wei<sup>1,*</sup> &nbsp; <a href="https://thuzhaowang.github.io/">Wang Zhao</a><sup>1</sup> &nbsp; <a href="https://czzzzh.github.io/">Chuhao Chen</a><sup>2</sup> &nbsp; <a href="https://amandaynzhou.github.io/">Yanning Zhou</a><sup>3,†</sup> &nbsp; <a href="https://yongjinliu.github.io/">Yong-Jin Liu</a><sup>1,†</sup></p>
<p align="center"><sup>1</sup>Tsinghua University &nbsp; <sup>2</sup>Carnegie Mellon University &nbsp; <sup>3</sup>The Chinese University of Hong Kong</p>
<p align="center"><sup>*</sup>Equal Contribution. &nbsp; <sup>†</sup>Corresponding Authors.</p>

<p align="center">
  <a href="https://hyzcluster.github.io/HairDeck"><img src="https://img.shields.io/badge/Project-Page-426b92" alt="项目主页"></a>
  <a href="https://arxiv.org/abs/2610.40599"><img src="https://img.shields.io/badge/arXiv-2610.40599-b31b1b" alt="arXiv: 2610.40599"></a>
  <a href="https://huggingface.co/hyz317/HairDeck"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Models-Hugging%20Face-ffd21e" alt="Hugging Face 模型权重"></a>
  <a href="https://huggingface.co/spaces/hyz317/HairDeck"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Demo-Try%20it-ffd21e" alt="体验在线演示"></a>
  <a href="#video"><img src="https://img.shields.io/badge/Video-Watch-426b92" alt="观看演示视频"></a>
</p>

<p align="center"><a href="README.md">English</a> · <strong>简体中文</strong> · <a href="README_ja.md">日本語</a></p>

<a name="video"></a>

## 演示视频

https://github.com/user-attachments/assets/3eec4de5-ed21-4b75-b7d7-4bd599cbfeb2

## 项目简介

**输入一张角色图，HairDeck 就能生成可编辑的三维发型，并配上驱动头发摆动的弹簧骨骼链。**

生成的发型由独立发片组成，每片都有自己的几何和 UV 坐标，方便继续逐片调整造型。生成结果可以导出为 GLB 网格，也可以绑定到角色后导出 VRM。

模型用 Card VAE 编码每片头发，以一个 token 表示一整片发片，再由 set diffusion Transformer 一起生成整套发型。chain diffusion model 根据生成的发片预测弹簧骨骼链，随后通过蒙皮完成绑定，让头发随角色动作摆动。

<p align="center">
  <img src="https://huggingface.co/hyz317/HairDeck/resolve/main/assets/method.png" width="100%" alt="HairDeck 方法：每片头发用一个 token 表示，根据图像或点云生成整套发型，再生成驱动头发的弹簧骨骼链。">
</p>

## 开始使用

[🛠️ 安装](#installation) · [🤗 模型权重](#model-weights) · [🚀 快速上手](#quick-start) · [🖼️ 本地演示](#local-demo) · [📦 输出文件](#outputs) · [📊 测试集评估](#test-set-evaluation) · [🏋️ 训练](#training) · [📝 引用](#citation)

<a name="installation"></a>

### 🛠️ 安装

运行需要 Linux x86-64、Python 3.12 和一张支持 CUDA 的 NVIDIA 显卡。我们在 PyTorch 2.8、CUDA 12.8 环境下做了测试。生成预览视频还需要安装 `ffmpeg`，并将它加入 `PATH`。

```bash
git clone https://github.com/hyz317/HairDeck.git
cd HairDeck

python -m pip install -r requirements.txt
python -m pip install --no-deps pyrender==0.1.45  # without its old PyOpenGL pin
```

后续命令都在仓库根目录下运行，并使用同一个 Python 环境。

<a name="model-weights"></a>

### 🤗 模型权重

模型权重和配套文件可以从 [hyz317/HairDeck](https://huggingface.co/hyz317/HairDeck) 下载。下载 SAM 3 和 DINOv3 前，需要先到 [facebook/sam3](https://huggingface.co/facebook/sam3) 和 [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) 页面接受许可条款，再登录 Hugging Face（国内可考虑从 ModelScope 下载）：

```bash
hf auth login
export HAIRDECK_ROOT=/path/to/workspace  # checkpoints, data, third-party weights and outputs
python scripts/setup/download_weights.py
```

脚本会下载 HairDeck 权重、图像预处理和编码所需的模型、配套数据，以及 Blender 4.2 LTS 和 VRM Add-on。Blender 默认下载 Linux x86-64 版本。

如果已安装 Blender 4.2 LTS 并启用了 VRM Add-on，将 `HAIRDECK_BLENDER` 设为它的启动命令即可。

默认工作目录结构如下：

```text
$HAIRDECK_ROOT/
├── ckpt/                       # HairDeck 模型权重和 latent 统计量
├── data/                       # 归一化统计量、角色模型和演示样例
└── third_party/
    ├── weights/
    │   ├── dinov3/             # facebook/dinov3-vitl16-pretrain-lvd1689m
    │   ├── sam3/               # facebook/sam3
    │   └── birefnet_matting/    # ZhengPeng7/BiRefNet-matting
    ├── dinov3_src/             # DINOv3 源码
    ├── sam3_src/               # SAM 3 源码
    └── blender/                # Blender 与 VRM Add-on
```

如果要修改模型、数据等文件的路径，可以通过环境变量或 JSON 文件设置。具体配置见 [configs/paths.example.json](configs/paths.example.json) 和 [hairdeck/paths.py](hairdeck/paths.py)。

<a name="quick-start"></a>

### 🚀 快速上手

下面的命令根据同一张角色图生成 4 个候选发型，再把第 0 个结果导出为 VRM 角色和转头预览视频：

```bash
python scripts/image_to_hair.py character.png --out results/my_character --n 4
python -m hairdeck.vrm.export results/my_character/character 0
```

结果按输入图片名保存到 `OUT/<image name>/`，上例对应 `results/my_character/character/`。

`--n` 指定生成数量（1–4），`--seed` 指定起始随机种子。加上 `--no-chains` 可以只生成发片。

如果使用其他模型权重，通过 `--ckpt` 和 `--chain_ckpt` 分别指定发片和骨骼链模型。两个模型需要基于同一个 Card VAE 训练。

<a name="local-demo"></a>

### 🖼️ 本地演示

```bash
python -m pip install -r requirements-demo.txt
python demo/app.py --port 7860
```

启动后，打开 [http://127.0.0.1:7860](http://127.0.0.1:7860)，就可以上传图片、查看生成的发型，并导出 VRM 或动画预览。也可以直接试用[在线演示](https://huggingface.co/spaces/hyz317/HairDeck)。

<a name="outputs"></a>

## 📦 输出文件

随机种子为 0 时，主要输出文件如下：

```text
results/my_character/character/
├── input.png                   # 输入图像
├── prep.png                    # 预处理掩码与裁剪结果
├── model_input.png             # 模型实际使用的裁剪图像
├── cards_seed0.json             # 发片几何数据
├── chains_seed0.json            # 生成的弹簧骨骼链
├── hair_seed0.glb               # 发型，含 head proxy、各发片的 UV 和发色
├── hair_seed0_nohead.glb         # 不含 head proxy 的发型
├── info.json                    # 生成参数和结果信息
├── avatar_seed0.vrm             # 由 VRM 导出命令生成
└── turn_seed0.mp4               # 由转头预览命令生成
```

其他结果的文件名按随机种子编号。使用 `--no-chains` 时不会生成骨骼链文件。

<a name="test-set-evaluation"></a>

## 📊 测试集评估

先下载评估所需的数据和模型，再运行点云和骨骼链两项测试。每项测试都会用随机种子 0、1、2 生成结果，打印论文中的评估指标，并将每个样本的结果和平均值写入 `eval.json`。

```bash
python scripts/setup/download_weights.py --only test_data charm_test_pc dora
python scripts/test.py pointcloud
python scripts/test.py chains                                # On the artists' cards
python scripts/test.py chains --cards results/pointcloud     # On the point-cloud model's cards
```

`results/pointcloud` 等评估目录默认位于 `HAIRDECK_ROOT` 下。如果修改了路径配置，评估会使用修改后的路径。

<a name="training"></a>

## 🏋️ 训练

准备好 CHARM 数据集和对应角色的 VRM 文件后，依次运行：

```bash
python scripts/setup/download_weights.py --only dora test_data charm_test_pc
python scripts/data.py DATASET_DIR VRM_DIR
bash scripts/train/launch_vae.sh
bash scripts/train/launch_pc_dit.sh && bash scripts/train/launch_img_dit.sh && bash scripts/train/launch_chain_dit.sh
```

训练结果保存在 `ckpt/train_*`，与下载的预训练权重分开存放。当前图像训练代码包含 synthetic appearance augmentation。已发布的图像模型还使用了一批经过风格化处理的渲染图像进行训练。

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

## 致谢

HairDeck 使用了 [CHARM](https://huggingface.co/datasets/hyz317/CHARM)、[Dora-VAE](https://huggingface.co/Seed3D/Dora-VAE-1.1)、[DINOv3](https://github.com/facebookresearch/dinov3)、[SAM 3](https://github.com/facebookresearch/sam3)、[BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet-matting) 和 [VRM Add-on for Blender](https://github.com/saturday06/VRM-Addon-for-Blender)。感谢以上项目的作者分享代码、模型和数据。

## 许可证

HairDeck 代码与模型权重采用 [Apache-2.0](LICENSE) 许可证。
第三方模型、代码和数据集遵循各自的许可证。
