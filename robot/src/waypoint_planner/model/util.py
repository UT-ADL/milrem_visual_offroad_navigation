import re

import torch

from gnm_train.models.gnm import GNM
from gnm_train.models.mdn import MDN
from waypoint_planner.model.vae import WaypointVAEGNM


def load_model(model_path, config):
    model_type = config["model_type"]
    if model_type == "vae":
        model = load_vae_model(model_path, config)
    elif model_type == "mdn":
        model = load_mdn_model(model_path, config)
    elif model_type == "gnm":
        model = load_gnm_model(model_path, config)
    elif model_type == "gnm-pretrained":
        model = load_gnm_model(model_path, config)
    else:
        raise Exception(f"Unknown model type {model_type}")

    return model


def load_gnm_model(model_path, config):
    model = GNM(config["context_size"],
                config["len_traj_pred"],
                config["learn_angle"],
                config["obs_encoding_size"],
                config["goal_encoding_size"])
    ckpt = torch.load(model_path, map_location="cuda:0")
    load_checkpoint(model, ckpt)
    return model


def load_vae_model(model_path, config):
    model = WaypointVAEGNM(config["context_size"],
                           config["obs_encoding_size"],
                           config["goal_encoding_size"],
                           config["latent_dim"],
                           config["len_traj_pred"],
                           config["learn_angle"])

    ckpt = torch.load(model_path)
    load_checkpoint(model, ckpt)
    return model


def load_mdn_model(model_path, config):
    model = MDN()
    ckpt = torch.load(model_path)
    load_checkpoint(model, ckpt)
    return model


def load_checkpoint(model, ckpt):
    if "state_dict" in ckpt:
        state_dict = ckpt['state_dict']
        # Remove prefix from keys
        state_dict = {re.sub(r'^model\.', '', k): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    else:  # For backwards compatibility with models trained with GNM repository code
        state_dict = ckpt['model'].state_dict()
        state_dict = {re.sub(r'^module\.', '', k): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)


def convert_to_onnx(input_model_path, output_model_path, config):
    model = WaypointVAEGNM(config["context_size"],
                            config["obs_encoding_size"],
                            config["goal_encoding_size"],
                            config["latent_dim"],
                            config["len_traj_pred"],
                            config["learn_angle"])

    if config['model_type'] == 'vae':
        def onnx_forward(self, obs_img: torch.tensor, goal_img: torch.tensor):
            predictions, _, _, _ = super().forward(obs_img, goal_img)
            sampled_predictions = self.sample(obs_img, 20)
            return predictions, sampled_predictions

        model.forward = onnx_forward

    model = load_model(input_model_path, config)
    model.eval()

    image_size = config["image_size"]
    obs_img_tensor = torch.randn(1, (config["context_size"]+1)*3, image_size[1], image_size[0], dtype=torch.float32)
    goal_img_tensor = torch.randn(1, 3, image_size[1], image_size[0], dtype=torch.float32)
    torch.onnx.export(model=model,
                      args=(obs_img_tensor, goal_img_tensor),
                      f=f"{output_model_path}",
                      opset_version=17,
                      input_names=['obs_img', 'goal_img']

                      )