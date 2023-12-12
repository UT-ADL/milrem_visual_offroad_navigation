## Overview

This repository contains reimplementation of [_ViKiNG: Vision-Based Kilometer-Scale Navigation with Geographic Hints_](https://sites.google.com/view/viking-release) 
article to control robot in offroad environment. Models can be pretrained with [_RECON dataset_](https://sites.google.com/view/recon-robot/home). 
Models can be also initialized using weights from [_GNM: A General Navigation Model to Drive Any Robot_](https://sites.google.com/view/drive-any-robot) project.

Directory Structure
------
    .
    ├── config                  # Configuration files
    ├── data                    # Data preprocessing, loading
    ├── gnm_train               # GNM model and training
    ├── models                  # VAE, MDN model definitions
    ├── notebooks               # Jupyter notebooks 
    ├── vint_train              # VinT and NoMaD model definitions  
    ├── viz                     # Visualizing model predictions  
    ├── train.py                # Python script for training local planner models 

## Environment

1. Install mamba to fasten the process. Or continue using conda if it is time for a coffee or two.

```bash
conda install mamba -c conda-forge
```

2. Create environment. Python version is very important. It has to be 3.9.16 as this works with both robotstack ROS and PyTorch.
You also have to make sure that it comes from conda's _defaults_ channel and not cpython version from _conda_forge_ channel,
otherwise Pytorch training is very slow.

```bash
conda create -n milrem-aire python=3.9.16
conda activate milrem-aire
```

3. Install ROS. You can skip this step if you already have ROS installed. See more from [robostack site](https://robostack.github.io/GettingStarted.html).

```bash
# Setup channels
conda config --env --add channels conda-forge
conda config --env --add channels robostack-staging
conda config --env --remove channels defaults

# Install ROS Noetic.
mamba install ros-noetic-desktop ros-noetic-image-geometry ros-noetic-ros-numpy numpy=1.23
```

4. Install PyTorch.
```bash
mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

5. Install other libraries.
```bash
mamba install pytorch-lightning tensorboard pandas jupyter matplotlib pytransform3d transformations h5py utm onnx opencv -c conda-forge
pip install onnxruntime-gpu
```

6. Install Weights and Biases and login.

```bash
pip install wandb
wandb login
```

Note that if you do not want to use wandb you can run `wandb offline` instead of login.

7. Copy config/sample-env.yaml to env.yaml and edit `dataset_root_path` to point the location where the dataset is located.
```bash
cp config/sample-env.yaml env.yaml
```

## Datasets

For training, datasets needs to be resampled with 4hz to the [GNM dataset format](https://github.com/robodhruv/drive-any-robot#data-processing). This provides common format and allows 
datasets to be used for training models using this repository, but also _Berkeley AI Research_ [GNM repository](https://github.com/PrieureDeSion/drive-any-robot) and
[ViNT/Nomad repository](https://github.com/robodhruv/visualnav-transformer).

### Milrem

Milrem dataset is located in neuron machine at `/data/milrem_aire/extraction/extracted_datasets/`. You can sync them to local machine using:
```bash
rsync -ah --info=progress2 username@neuron.hpc.ut.ee:/data/milrem_aire/extraction/extracted_datasets/ .
```

For training dataset need to be processed to GNM format. During this process also bad data is filtered as defined [here](data/milrem_tracks.py).

```bash
python -m data.process_milrem --input-dir <extracted Milrem dataset> --output-dir <target directory>
```

Use following command to create train/test data split. Split is defined statically [tracks definition](data/milrem_tracks.py).

```bash
python -m data.milrem_data_split
```

#### Adding new track

Do add new track to the dataset, it has to be added to the list of [track definitions](data/milrem_tracks.py) and 
both data script above must be runned again.

### RECON

Download RECON dataset from [here](https://rail.eecs.berkeley.edu/datasets/recon-navigation/recon_dataset.tar.gz) and preprocess:

```bash
wget https://rail.eecs.berkeley.edu/datasets/recon-navigation/recon_dataset.tar.gz
tar -xvf recon_dataset.tar.gz

python -m data.process_recon --input-dir <extracted RECON dataset> --output-dir <target directory>
```

```bash
python -m data.recon_data_split --data-dir <path to RECON data> --dataset-name recon
```

## Training

There is support for training following model architectures:

- Variational autoencoder (VAE)
- General Navigation Model (GNM)
- Mixture Desity Network (MDN)

Train model:
```bash
python train.py --config config/<model type>.yaml
```

Use `--config` to defined model configuration. For example `config/vae.yaml`, `config/gnm.yaml`, `config/mdn.yaml`.

Use `--checkpoint` parameter to continue training from previous checkpoint. Pretrained GNM weights can be downloaded 
from [here](https://drive.google.com/drive/folders/1np7D0Ak7x10IoQn9h0qxn8eoxJQiw8Dr).

### Wandb

Training will also be logged into Weights & Biases.

To disable syncing log to the cloud run:
```bash
wanbd offline
```

To enable cloud logging, just run:
```bash
wanbd online
```

### Tensorboard

Alternatively training can be monitored by running tensor board:
```bash
tensorboard --logdir logs
```

## Visualize

Model predictions can be visualized off-policy using dataset using following command:
```bash
python -m viz.visualizer --dataset-path {path to dataset track} --model-path {path to model checkpoint} --model-config config/vae.yaml
```

Or with model in ONNX format:
```bash
python -m viz.visualizer --dataset-path {path to dataset track} --model-path {path to ONNX model} --onnx --model-config config/vae.yaml
```

Use `--output-file` parameter to specify output video file name. In this case video file is generated instead of interactive display.

Use `--dataset-path` parameter to specify location of the dataset track to be used for visualization. 
For example _ut/milrem/data/2023-05-18-16-57-00_. Note that this has to point to unprocessed dataset as it needs every
frame and processed dataset is resampled.

Use `--model-path` to specify location of the model to be used for making predictions. 
For example _logs/GNM/version_8/checkpoints/last.ckpt_. For pretrained GNM model path to pretrained weights in _pth_ format can be used.

Use `--model-config` to specify model configuration. For example _config/vae.yaml_.

Use `--onnx` flag if model is saved in ONNX format.

Use `--goal-conditioning` to condition model using short distance goals. If excluded model is in exploration mode and
last image of the dataset is used as goal.

Use `--start-frame` parameter to specify the frame visualization starts from.

Use `u`, `i` keys to move visualization 1 frame backwards or forwards respectively.

Use `j`, `k` keys to move visualization 30 frames backwards or forwards respectively.

## Export to ONNX

Download pretrained ViNT and NoMaD models from [here](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg).

ViNT model:
```bash
python convert_pytorch_to_onnx.py --model-config config/vint.yaml -i <path to vint.pth> -o <output file name>
```

NoMaD model:
```bash
python convert_pytorch_to_onnx.py --model-config config/nomad.yaml -i <path to nomad.pth> -o <output folder>
```
Unlike other models, NoMaD model is split into 3 models (`action.onnx`, `distance.onnx`, `encoder.onnx`) as converting
to just one results in diffusion part being very slow.

These models can then be used with `NomadOnnx` class, which handles the diffusion step and other details:
```bash
from model.nomad_onnx import NomadOnnx
import numpy as np

nomad_onnx = NomadOnnx("model_weights/nomad")

# replace with real input
goal_tensor = np.random.randn(1, 3, 96, 96).astype(np.float32)
obs_tensor = np.random.randn(1, 12, 96, 96).astype(np.float32)

predicted_dist, predicted_actions = nomad_onnx.predict(obs_tensor, goal_tensor)
```