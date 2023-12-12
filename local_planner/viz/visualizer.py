import argparse
from pathlib import Path

import yaml

from data.dataset import MilremVizDataset
from model.util import load_model
from viz.nomad_onnx_visualizer import NomadOnnxVisualizer
from viz.vae_visualizer import VAEVisualizer
from viz.onnx_visualizer import OnnxVisualizer
from viz.gnm_visualizer import GNMVisualizer
from viz.nomad_visualizer import NomadVisualizer


def parse_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--dataset-path',
        required=True
    )
    argparser.add_argument(
        '--model-config',
        required=True
    )
    argparser.add_argument(
        '--model-path',
        required=True
    )
    argparser.add_argument(
        '--onnx',
        required=False,
        action='store_true',
        help="Whether to use model is in onnx format instead of PyTorch."
    )
    argparser.add_argument(
        '--output-file',
        '-o',
        required=False,
        help="Output video file name."
    )
    argparser.add_argument(
        '--start-frame',
        required=False,
        default=0,
        type=int,
        help="Dataset location to start from creating the video"
    )
    argparser.add_argument(
        '--goal-conditioning',
        required=False,
        action='store_true',
        help="Whether to use goal image or last frame of the track."
    )
    return argparser.parse_args()


def create_visualizer(model_type, model_path, goal_conditioning=True):
    if model_type == "vae":
        return VAEVisualizer(goal_conditioning)
    elif model_type in ["gnm", "gnm-pretrained", "vint"]:
        return GNMVisualizer(goal_conditioning)
    elif model_type == "nomad":
        return NomadVisualizer(goal_conditioning)
    elif model_type == "onnx":
        return OnnxVisualizer(goal_conditioning)
    elif model_type == "nomad-onnx":
        return NomadOnnxVisualizer(model_path, goal_conditioning)
    else:
        raise Exception(f"Unknown model type: {model_type}")


def load_dataset(dataset_path, config):
    return MilremVizDataset(dataset_path, **config)


if __name__ == "__main__":
    args = parse_arguments()
    dataset_path = Path(args.dataset_path)
    model_config = args.model_config
    model_path = args.model_path
    is_onnx = args.onnx
    output_file = args.output_file
    start_frame = args.start_frame
    goal_conditioning = args.goal_conditioning

    with open(model_config, 'r') as file:
        config = yaml.safe_load(file)

    if is_onnx:
        # There is no model to load, onnx inference session is created inside the visualizer
        model = model_path
        model_type = "nomad-onnx" if config['model_type'] == "nomad" else "onnx"
    else:
        model = load_model(model_path, config)
        model.eval()
        model_type = config['model_type']

    dataset = load_dataset(dataset_path, config)
    viz = create_visualizer(model_type, model_path, goal_conditioning)

    if output_file:
        viz.create_video(dataset, model, output_file, start_frame)
    else:
        viz.create_interactive(dataset, model, start_frame)
