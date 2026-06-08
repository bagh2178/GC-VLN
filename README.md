# GC-VLN: Instruction as Graph Constraints for Training-free Vision-and-Language Navigation

### [Paper](https://arxiv.org/abs/2509.10454) | [Project Page](https://bagh2178.github.io/GC-VLN/) | [Video](https://cloud.tsinghua.edu.cn/f/f21a13df2bc749bb980a/?dl=1)

> GC-VLN: Instruction as Graph Constraints for Training-free Vision-and-Language Navigation  
> [Hang Yin](https://bagh2178.github.io/)*, [Haoyu Wei](https://Why-peace.github.io/)*, [Xiuwei Xu](https://xuxw98.github.io/)$\dagger$, [Wenxuan Guo](https://gwxuan.github.io/), [Jie Zhou](https://scholar.google.com/citations?user=6a79aPwAAAAJ&hl=en&authuser=1), [Jiwen Lu](http://ivg.au.tsinghua.edu.cn/Jiwen_Lu/)$\ddagger$

\* Equal contribution $\dagger$ Project leader $\ddagger$ Corresponding author

We propose a unified 3D graph representation for <b>zero-shot</b> vision-and-language navigation. By modeling instruction graph as constraints, we can solve the optimal navigation path accordingly. Wrong exploration can also be handled by graph-based backtracking.

## News
- [2026/06/08]: Release code.
- [2025/09/16]: Arxiv and project page available.
- [2025/08/01]: GC-VLN is accepted to CoRL 2025!

## Demo
![demo](./assets/demo.gif)

## Method

Method Pipeline:
![Overview](./assets/pipeline.png)

## Installation

### Step 1: Clone the repository

```bash
git clone --recursive https://github.com/bagh2178/GC-VLN.git
cd GC-VLN
```

### Step 2: Create and activate GC-VLN environment

```bash
conda create -n GC-VLN python=3.9
conda activate GC-VLN
conda install habitat-sim==0.2.4 -c conda-forge -c aihabitat
pip install -e third_party/habitat-lab
pip install -r requirements.txt
conda install faiss-gpu=1.8.0 -c pytorch -y
pip install --no-build-isolation -e third_party/GLIP
mkdir -p third_party/GLIP/MODEL
wget -O third_party/GLIP/MODEL/glip_large_model.pth https://huggingface.co/GLIPModel/GLIP/resolve/main/glip_large_model.pth?download=true
pip install -e third_party/ModelServer
pip install git+https://github.com/facebookresearch/pytorch3d.git --no-build-isolation
```

### Step 3: Create GC-VLN-Server environment

```bash
conda create -n GC-VLN-Server python=3.10
conda activate GC-VLN-Server
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install opencv-python==4.11.0.86 supervision==0.25.1 transformers==4.51.3 addict==2.4.0 yapf==0.43.0 pycocotools==2.0.8 timm==1.0.15 numpy==2.2.4  # supervision addict yapf pycocotools
pip install -e third_party/GC-VLN-Server
pip install -e third_party/Grounded-SAM-2
conda install intel-openmp=2021.4.0 -c defaults -y
pip install --no-build-isolation -e third_party/Grounded-SAM-2/grounding_dino
bash third_party/Grounded-SAM-2/checkpoints/download_ckpts.sh
bash third_party/Grounded-SAM-2/gdino_checkpoints/download_ckpts.sh
```

## Dataset

We use [R2R-CE](https://github.com/jacobkrantz/VLN-CE), [RxR-CE](https://github.com/jacobkrantz/VLN-CE) datasets, and [Matterport3D (MP3D)](https://niessner.github.io/Matterport/) scene data. The dataset structure should be organized as follows:

```
GC-VLN/
└── data/
    ├── datasets/
    │   ├── R2R_VLNCE_v1-2_preprocessed/
    │   │   └── val_unseen/
    │   │       └── val_unseen.json.gz
    │   └── RxR_VLNCE_v0/
    │       └── val_unseen/
    │           └── val_unseen_guide.json.gz
    └── scene_datasets/
        └── mp3d/
            ├── 1LXtFkjw3qL/
            │   ├── 1LXtFkjw3qL.glb
            │   ├── 1LXtFkjw3qL.house
            │   ├── 1LXtFkjw3qL.navmesh
            │   └── 1LXtFkjw3qL_semantic.ply
            ├── 1pXnuDYAj8r/
            ├── ...
            └── zsNo4HB9uLZ/
```

## Evaluation

### Step 1: Start the GSAM2 GC-VLN-Server

Activate GC-VLN-Server environment and start the GSAM2 server:

```bash
conda activate GC-VLN-Server
python third_party/GC-VLN-Server/scripts/quickstart_server/GSAM2.py --port 7000
```

Wait for the server to fully start before proceeding to the next step.

### Step 2: Run VLN Evaluation

In a new terminal, activate GC-VLN environment and run evaluation:

```bash
conda activate GC-VLN

# For R2R dataset
bash run_eval.sh r2r

# For RxR dataset
bash run_eval.sh rxr
```

## Code Structure

```
GC-VLN/
├── src/
│   ├── solver/                    # Navigation planning and constraint solving
│   ├── scenegraph/                # Scene graph construction and mapping
│   ├── agent/                     # Agent and environment wrappers
│   └── habitat_extensions/        # Custom Habitat components
├── third_party/
│   ├── ModelServer/               # GSAM2 model server for segmentation
│   ├── Grounded-SAM-2/            # Grounded-SAM-2 implementation
│   ├── habitat-lab/               # Habitat simulation platform
│   └── GLIP/                      # Grounded Language-Image Pretraining model
├── config/                      # Configuration files
├── data/                        # Data directory (datasets, scene_datasets)
├── outputs/                     # Output directory for logs and results
└── main.py                      # Evaluation entry point
```

## Relevant Work
Check out our scene graph-based zero-shot navigation series:
- [SG-Nav](https://bagh2178.github.io/SG-Nav/) for zero-shot object-goal navigation.
- [UniGoal](https://github.com/bagh2178/UniGoal) for zero-shot goal-oriented navigation.

## Citation

```bibtex
@article{yin2025gcvln,
      title={GC-VLN: Instruction as Graph Constraints for Training-free Vision-and-Language Navigation},
      author={Hang Yin and Haoyu Wei and Xiuwei Xu and Wenxuan Guo and Jie Zhou and Jiwen Lu},
      journal={arXiv preprint arXiv:2509.10454},
      year={2025}
}
```

