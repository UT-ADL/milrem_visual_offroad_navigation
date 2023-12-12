import re

import torch

from models.heuristic import SegmentiveHeuristic


def convert_to_onnx(input_model_path, output_model_path, config):
    model = load_model(input_model_path)
    model.eval()

    map_size = config["map_size"]
    masked_map = torch.randn(1, 5, 2 * map_size, 2 * map_size, dtype=torch.float32)
    torch.onnx.export(model=model,
                      args=(masked_map),
                      f=output_model_path,
                      opset_version=17,
                      input_names=['masked_map'])


def load_model(model_path):
    model = SegmentiveHeuristic(n_channels=5)
    ckpt = torch.load(model_path, map_location="cuda:0")
    state_dict = ckpt['state_dict']
    state_dict = {re.sub(r'^model\.', '', k): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model