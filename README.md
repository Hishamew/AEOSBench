# Constellation

This repository is the official implementation of "Towards Realistic Earth-Observation Constellation Scheduling: Benchmark and Methodology".

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-purple)](https://neurips.cc/virtual/2025/loc/san-diego/poster/116515)
[![arXiv](https://img.shields.io/badge/arXiv-2510.26297-b31b1b.svg)](https://arxiv.org/abs/2510.26297)

## Install Miniconda (recommended)

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
~/miniconda3/bin/conda init
```

Create and activate an environment:

```bash
conda create -n aeosbench python=3.11.10 -y
conda activate aeosbench
```

## Installation

### venv method

```bash

sudo apt install ffmpeg libpq-dev
bash setup.sh
```

### Conda method

If you don't want to use venv, conda is also welcomed：

```bash
conda env create --name aeosbench -f environment.yaml
conda activate aeosbench
```

## Data Preparation

### Download Benchmark

You can just download the val_seen/val_unseen/test from the trajectories inside our hf repo and unzip them to only evaluate your own model:

```bash
git clone -b eval git@hf.co:datasets/MessianX/AEOS-dataset ./data
find ./data -type f -name '*.tar' -print0 | xargs -0 -n1 -I{} sh -c 'tar -xf "$1" -C "$(dirname "$1")"' _ {}
```

### Download Training Data

If you want to use all data to reproduce our paper:

```bash
git clone -b main git@hf.co:datasets/MessianX/AEOS-dataset ./data
find ./data -type f -name '*.tar' -print0 | xargs -0 -n1 -I{} sh -c 'tar -xf "$1" -C "$(dirname "$1")"' _ {}
```

### Generate Training Data

This is an optional step.

First, use the script to generate satellites.

```bash
./tools/generate_satellites.sh
```

After generated enough satellites, use them to generate constellations, together with enough tasksets.

```bash
python tools/generate_constellations_and_tasksets.py
```

Then do iterative simulations and generate trajectories.

```bash
torchrun --nproc-per-node 50 tools/generate_trajectories.py
mv data/trajectories data/trajectories.0
torchrun --nproc-per-node 50 tools/generate_trajectories.py --previous-trajectories data/trajectories.0
mv data/trajectories data/trajectories.1
torchrun --nproc-per-node 50 tools/generate_trajectories.py --previous-trajectories data/trajectories.0 data/trajectories.1
mv data/trajectories data/trajectories.2
torchrun --nproc-per-node 50 tools/generate_trajectories.py --previous-trajectories data/trajectories.0 data/trajectories.1 data/trajectories.2
# ... as many as you want
```

Finally, use this to generate annotations:

```bash
python ./tools/compare_trajectory_cr.py
```

### Expected Data Organization

The file tree should be like this：

```text
data/
├── satellites/
│   ├── {train,val_seen}/
│   │   └── {0..9999}.json
│   ├── val_unseen/
│   │   └── {0..1999}.json
│   └── test/
│       └── {0..1999}.json
├── orbits/
│   └── {0..6884}.json
├── constellations/
│   ├── train/
│   │   └── {00..99}/
│   │       └── {00000..99999}.json
│   └── {test,val_seen,val_unseen}/
│       └── 00/
│           └── {00000..00999}.json
└── tasksets/
│   ├── train/
│   │   └── {00..99}/
│   │       └── {00000..99999}.json
│   └── {test,val_seen,val_unseen}/
│       └── 00/
│           └── {00000..00999}.json
├── trajectories.1/
│   ├── train/
│   │   └── {00..25}/
│   │       ├── {00000..25999}.json
│   │       └── {00000..25999}.pth
│   └── {test,val_seen,val_unseen}/
│       └── 00/
│           ├── {00000..00999}.json
│           └── {00000..00999}.pth
├── trajectories.2/
│   └── train/
│       └── {00..25}/
│           ├── {00000..25999}.json
│           └── {00000..25999}.pth
├── trajectories.3/
│   └── train/
│       └── {00..25}/
│           ├── {00000..25999}.json
│           └── {00000..25999}.pth
└── annotations/
    ├── test.json
    ├── train.json
    ├── val_seen.json
    └── val_unseen.json
```

Note: The ids used inside different split(test/train/val_seen/val_unseen) can be discontinuous.

## Training

Use the command below to train our model:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=:${PYTHONPATH} auto_torchrun -m constellation.new_transformers.train test constellation/new_transformers/config.py
```

This will continue till 200000 iters.

## Evaluation

Use the command below to evaluate the model:

```bash
CUDA_VISIBLE_DEVICES=0 WORLD_SIZE=1 RANK=0 python -m constellation.rl.eval_all \
    work_dir_name \
    constellation/rl/config_eval.py \
    --load-model-from 'work_dirs/test/checkpoints/iter_100000/model.pth'
```
