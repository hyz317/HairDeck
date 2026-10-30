<h1 align="center">HairDeck: Animation-Ready Stylized Hair Cards<br>from a Single Image</h1>

<p align="center"><a href="https://hyzcluster.github.io/">Yuze He</a><sup>1,*</sup> &nbsp; Yixiang Wei<sup>1,*</sup> &nbsp; <a href="https://thuzhaowang.github.io/">Wang Zhao</a><sup>1</sup> &nbsp; <a href="https://czzzzh.github.io/">Chuhao Chen</a><sup>2</sup> &nbsp; <a href="https://amandaynzhou.github.io/">Yanning Zhou</a><sup>3,†</sup> &nbsp; <a href="https://yongjinliu.github.io/">Yong-Jin Liu</a><sup>1,†</sup></p>
<p align="center"><sup>1</sup>Tsinghua University &nbsp; <sup>2</sup>Carnegie Mellon University &nbsp; <sup>3</sup>The Chinese University of Hong Kong</p>
<p align="center"><sup>*</sup>Equal contribution. &nbsp; <sup>†</sup>Corresponding authors.</p>

<p align="center">
  <a href="https://hyzcluster.github.io/HairDeck"><img src="https://img.shields.io/badge/Project-Page-426b92" alt="Project page"></a>
  <a href="https://arxiv.org/abs/2610.40599"><img src="https://img.shields.io/badge/arXiv-2610.40599-b31b1b" alt="arXiv: 2610.40599"></a>
  <a href="https://huggingface.co/hyz317/HairDeck"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Models-Hugging%20Face-ffd21e" alt="Model weights on Hugging Face"></a>
  <a href="https://huggingface.co/spaces/hyz317/HairDeck"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Demo-Try%20it-ffd21e" alt="Try the online demo"></a>
  <a href="#video"><img src="https://img.shields.io/badge/Video-Watch-426b92" alt="Watch the demonstration"></a>
</p>

## Video

https://github.com/user-attachments/assets/3eec4de5-ed21-4b75-b7d7-4bd599cbfeb2

## Overview

**HairDeck generates stylized hair cards and the spring-bone chains that animate them from a single image.**
Each hair card is represented by one token, and a set diffusion transformer generates the cards jointly.
The generated hairstyle can be exported as a GLB mesh or attached to an avatar and exported as VRM.

- **Hair cards as a set.** A card VAE encodes each card, and conditional set diffusion generates the hairstyle as a whole.
- **Geometry with UVs.** The output preserves individual hair cards and provides per-card UV coordinates.
- **Spring-bone animation.** A chain diffusion model generates the bones that drive the cards, with skinning and VRM export.

<p align="center">
  <img src="https://huggingface.co/hyz317/HairDeck/resolve/main/assets/method.png" width="100%" alt="HairDeck pipeline: hair-card tokens, image- or point-cloud-conditioned set diffusion, and generated cards with spring-bone chains.">
</p>

## Getting started

[🛠️ Installation](#installation) · [🤗 Model weights](#model-weights) · [🚀 Quick start](#quick-start) · [🖼️ Local demo](#local-demo) · [📦 Outputs](#outputs) · [📊 Evaluation](#test-set-evaluation) · [🏋️ Training](#training) · [📝 Citation](#citation)

<a name="installation"></a>

### 🛠️ Installation

Use Linux x86-64, Python 3.12 and one NVIDIA CUDA GPU. The release was tested with PyTorch 2.8 and CUDA 12.8.
The video preview also needs `ffmpeg` on your `PATH`.

```bash
git clone https://github.com/hyz317/HairDeck.git
cd HairDeck

python -m pip install -r requirements.txt
python -m pip install --no-deps pyrender==0.1.45  # without its old PyOpenGL pin
```

Run the commands below from the repository root, in the same Python environment.

<a name="model-weights"></a>

### 🤗 Model weights

Our checkpoints and supporting assets are hosted at [hyz317/HairDeck](https://huggingface.co/hyz317/HairDeck).
SAM 3 and DINOv3 are gated: first accept their licences on [facebook/sam3](https://huggingface.co/facebook/sam3)
and [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m), then log in:

```bash
hf auth login
export HAIRDECK_ROOT=/path/to/workspace  # checkpoints, data, third-party weights and outputs
python scripts/setup/download_weights.py
```

The downloader sets up the HairDeck checkpoints, preprocessing and image-encoder weights, supporting data,
and Blender 4.2 LTS with the VRM Add-on. The default Blender download is for Linux x86-64.
To use an existing Blender 4.2 LTS installation with the VRM Add-on enabled, set `HAIRDECK_BLENDER` to its command.

The default workspace layout is:

```text
$HAIRDECK_ROOT/
├── ckpt/                       # HairDeck checkpoints and latent statistics
├── data/                       # Normalization statistics, avatar and demo examples
└── third_party/
    ├── weights/
    │   ├── dinov3/             # facebook/dinov3-vitl16-pretrain-lvd1689m
    │   ├── sam3/               # facebook/sam3
    │   └── birefnet_matting/    # ZhengPeng7/BiRefNet-matting
    ├── dinov3_src/             # DINOv3 source
    ├── sam3_src/               # SAM 3 source
    └── blender/                # Blender and the VRM Add-on
```

Individual locations can be configured through environment variables or a JSON file; see
[configs/paths.example.json](configs/paths.example.json) and [hairdeck/paths.py](hairdeck/paths.py).

<a name="quick-start"></a>

### 🚀 Quick start

Generate four variations from a character image, then export variation 0 as a VRM avatar and a head-turn video:

```bash
python scripts/image_to_hair.py character.png --out results/my_character --n 4
python -m hairdeck.vrm.export results/my_character/character 0
```

Results are written to `OUT/<image name>/`, here `results/my_character/character/`.
`--n` accepts 1–4 variations; `--seed` sets the first seed. Use `--no-chains` to generate cards only.
For custom models, pass `--ckpt` and a matching `--chain_ckpt` trained with the same card VAE.

<a name="local-demo"></a>

### 🖼️ Local demo

```bash
python -m pip install -r requirements-demo.txt
python demo/app.py --port 7860
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860) to generate hair, inspect variations, and export a VRM or animation preview.
You can also try the [hosted demo](https://huggingface.co/spaces/hyz317/HairDeck).

<a name="outputs"></a>

## 📦 Outputs

The main files for seed 0 are:

```text
results/my_character/character/
├── input.png                   # Input image
├── prep.png                    # Preprocessing masks and crop
├── model_input.png             # Image crop used by the model
├── cards_seed0.json             # Hair-card geometry
├── chains_seed0.json            # Generated spring-bone chains
├── hair_seed0.glb               # Hair with a head proxy, per-card UVs and hair color
├── hair_seed0_nohead.glb         # Hair without the head proxy
├── info.json                    # Generation settings and result metadata
├── avatar_seed0.vrm             # Created by the VRM export command
└── turn_seed0.mp4               # Created by the head-turn preview command
```

Other variations use their corresponding seed in the filename. The chains file is omitted with `--no-chains`.

<a name="test-set-evaluation"></a>

## 📊 Test-set evaluation

Download the evaluation assets, then run the point-cloud and chain tests. Each test samples seeds 0, 1 and 2,
prints the corresponding paper metrics, and saves per-sample results and their means in `eval.json`.

```bash
python scripts/setup/download_weights.py --only test_data charm_test_pc dora
python scripts/test.py pointcloud
python scripts/test.py chains                                # On the artists' cards
python scripts/test.py chains --cards results/pointcloud     # On the point-cloud model's cards
```

Evaluation paths such as `results/pointcloud` follow the workspace layout under `HAIRDECK_ROOT` and its path overrides.

<a name="training"></a>

## 🏋️ Training

Start from the CHARM dataset and the characters' VRM files:

```bash
python scripts/setup/download_weights.py --only dora test_data charm_test_pc
python scripts/data.py DATASET_DIR VRM_DIR
bash scripts/train/launch_vae.sh
bash scripts/train/launch_pc_dit.sh && bash scripts/train/launch_img_dit.sh && bash scripts/train/launch_chain_dit.sh
```

The launchers write to `ckpt/train_*`, separately from the downloaded checkpoints. The current image-training
recipe includes synthetic appearance augmentation; the restyled-render pool used for the released image model
is not included in this release.

<a name="citation"></a>

## 📝 Citation

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

## Acknowledgments

HairDeck uses [CHARM](https://huggingface.co/datasets/hyz317/CHARM), [Dora-VAE](https://huggingface.co/Seed3D/Dora-VAE-1.1), [DINOv3](https://github.com/facebookresearch/dinov3), [SAM 3](https://github.com/facebookresearch/sam3), [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet-matting), and the [VRM Add-on for Blender](https://github.com/saturday06/VRM-Addon-for-Blender). We thank their authors for making these resources available.

## License

HairDeck code and checkpoints are licensed under [Apache-2.0](LICENSE).
Third-party models, code and datasets retain their respective licences.
