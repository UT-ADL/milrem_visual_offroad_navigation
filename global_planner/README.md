## Overview

This repository contains global planner implemention of [_ViKiNG: Vision-Based Kilometer-Scale Navigation with Geographic Hints_](https://sites.google.com/view/viking-release) 
article to control robot in offroad environment.

Directory Structure
------
    .
    ├── config                  # Configuration files
    ├── data                    # All data related code: dataset processing, mapping, etc
    ├── model_weights           # Pretrained model weights
    ├── models                  # Model definitions
    ├── notebooks               # Jupyter notebooks 
    ├── viz                     # Visualizing model predictions 
    ├── train_contrastive.py    # Python script for training global planner using contrastive learing 
    ├── train_segment.py        # Python script for training global planner using segmentation base solution 

## Environment


### Without PyTorch

To test models converted to ONNX, simplified environment can be used.

```bash
conda create -n global-planner-onnx python=3.9 pandas matplotlib jupyter pyyaml opencv rasterio pyproj
conda activate global-planner-onnx
pip install onnxruntime-gpu  # Install onnxruntime instead if no GPU
```

Alternatively environment can be created from exported environment configuration:
```bash
conda env create -f environment-onnx.yml
```

Environment can be tested by running notebook `notebooks/how-to-use-compact.ipynb` and checking the output.

### With PyTorch

To also train models, Pytorch and other libraries must be installed.

```bash
conda create -n global-planner-torch pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia 
conda activate global-planner-torch
conda install pytorch-lightning tensorboard pandas matplotlib jupyter pyyaml opencv rasterio pyproj onnx -c conda-forge
pip install wandb onnxruntime-gpu
```

Alternatively environment can be created from exported environment configuration:
```bash
conda env create -f environment-torch.yml
```


## Data

### Maps

There is 7 different type of maps that can be used as geographical hints in global planner model:
- base
- baseelev
- hybrid
- orienteering
- ortho
- roadmap
- satellite

`base`, `baseelev` and `orienteering` maps are performing the best with current implementations.

Map files are located in neuron machine at `/data/milrem_aire/maps/utm`. You can sync them to local machine using:
```bash
rsync -ah --info=progress2 username@neuron.hpc.ut.ee:/data/milrem_aire/maps/utm .
```

### Trajectories

Trajectories are used from orienteering dataset located in neuron machine at `/data/milrem_aire/extraction/extracted_datasets/`. You can sync them to local machine using:
```bash
rsync -ah --info=progress2 username@neuron.hpc.ut.ee:/data/milrem_aire/extraction/extracted_datasets/ .
```

Note that if you don't train local planner, only metadata from csv files is needed for global planner and images does not need to be synced.

## Training

Copy environment specific configuration `config/sample-env.yaml` to `env.yaml` and edit dataset and map paths to point to the location where these are located.
```bash
cp config/sample-env.yaml env.yaml
```

Configuration file `neuron-env.yaml` can be used for training on `neuron.hpc.ut.ee` machine. 

This environment specific configuration file can be used to also override parameters (`batch-size`, `max-epochs`, etc) 
in the main training configuration file.

### Segmentation based models

Segmentation based model uses different approach compared to original article, but works better.

To train model with segmentation based approach use following command:

```bash
python train_segment.py --config config/default_segment.yaml --run-name descriptive-name
```

Use `--config` parameter to specify training configuration used.

### Contrastive Learning based models

This solution uses similar approach as was done in the original paper, but current implementation doesn't work well.
Model seems to learn mostly direct routes from current position to the goal position and ignore geographical hints from
the input map.

To train model with contrastive learning based approach use following command:

```bash
python train_contrastive.py --config config/default_contrastive.yaml --run-name descriptive-name
```

Use `--config` parameter to specify training configuration used.

## Testing

Models can be tested either doing off-policy prediction or using simulator to imitate on-policy behaviour.

### Off-policy predictions

To create off-policy predictions video using segmentation based model, run following command:

```bash
python -m viz.create_segment_video --config config/default_segment.yaml --dataset-name 2023-07-27-16-12-51 --model-path model_weights/distance_segment.ckpt -o output.mp4
```

Use `--config` parameter to specify training configuration used.

Use `--dataset-name` parameter to specify dataset used. `dataset_path` must be defined in `env.yaml`, see `sample-env.yaml` for sample configration. 

Use `--model-path` parameter to specify model used.

Use `--output-file` optional parameter to specify output video file. Interactive mode is used if excluded.

Off-policy predictions video for contrastive learning models can be made similarily:

```bash
python -m viz.create_contrastive_video --config config/default_contrastive.yaml --dataset-name 2023-05-11-17-08-21 --model-path model_weights/contrastive.ckpt -o output-contrast.mp4
```

### On-policy simulation
```bash
python -m viz.simulator -c config/default_segment.yaml --dataset-name 2023-08-25-15-48-47 --model-path model_weights/distance_segment.onnx 
```

Use `--config-path` parameter to specify configuration.

Use `--dataset-name` parameter to specify dataset used. `dataset_path` must be defined in `env.yaml`, see `sample-env.yaml` for sample configration. 

Use `--model-path` parameter to specify model used.

On-policy simulation only support segmentation based models currently.
