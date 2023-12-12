import re
from pathlib import Path

import torch
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from torch import nn

from gnm_train.models.mdn import MDN
from model.gnm import GNM
from model.nomad_torch import NoMaDDiffuser
from model.vae import WaypointVAEGNM
from vint_train.models.vint.vint import ViNT


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
    elif model_type == "vint":
        model = load_vint_model(model_path, config)
    elif model_type == "nomad":
        model = load_nomad_model(model_path, config)
    else:
        raise Exception(f"Unknown model type {model_type}")

    return model


def load_vint_model(model_path, config):

    model = ViNT(
        context_size=config["context_size"],
        len_traj_pred=config["len_traj_pred"],
        learn_angle=config["learn_angle"],
        obs_encoder=config["obs_encoder"],
        obs_encoding_size=config["obs_encoding_size"],
        late_fusion=config["late_fusion"],
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
    )

    ckpt = torch.load(model_path)
    load_checkpoint(model, ckpt)

    return model


def load_nomad_model(model_path, config):
    nomad = NoMaDDiffuser(config)

    ckpt = torch.load(model_path)
    load_checkpoint(nomad, ckpt)

    return nomad


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
    elif "model" in ckpt:
        state_dict = ckpt['model'].state_dict()
        state_dict = {re.sub(r'^module\.', '', k): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(ckpt)


def convert_to_onnx(input_model_path, output_model_path, config):

    model = load_model(input_model_path, config)
    model.eval()

    if config['model_type'] == 'vae':
        def onnx_forward(self, obs_img: torch.tensor, goal_img: torch.tensor):
            predictions, _, _, _ = super().forward(obs_img, goal_img)
            sampled_predictions = self.sample(obs_img, 20)
            return predictions, sampled_predictions

        model.forward = onnx_forward

    image_size = config["image_size"]
    obs_img_tensor = torch.randn(1, (config["context_size"] + 1) * 3, image_size[1], image_size[0], dtype=torch.float32)
    goal_img_tensor = torch.randn(1, 3, image_size[1], image_size[0], dtype=torch.float32)

    input_tensor = [obs_img_tensor, goal_img_tensor]

    if config['model_type'] == 'nomad':
        # Create directory for models, fail if exists
        output_folder = Path(output_model_path)
        output_folder.mkdir(parents=True, exist_ok=False)

        # NoMaD model has to broken up as diffusion step does has poor performance if converted to ONNX
        mask = torch.zeros(1).long().to(obs_img_tensor.device)
        obs_img_tensor = obs_img_tensor.repeat(goal_img_tensor.shape[0], 1, 1, 1)
        mask = mask.repeat(goal_img_tensor.shape[0])
        input_tensor = [obs_img_tensor, goal_img_tensor, mask]

        # Vision features encoder
        torch.onnx.export(model=model.vision_encoder,
                          args=tuple(input_tensor),
                          f=f"{output_model_path}/encoder.onnx",
                          opset_version=17,
                          input_names=['obs_img', 'goal_img', 'input_goal_mask'])

        num_samples = config["num_samples"]
        len_traj_pred = config["len_traj_pred"]
        encoding_size = config["encoding_size"]
        vision_features = torch.randn(num_samples, encoding_size)

        # Distance predictor
        torch.onnx.export(model=model.dist_pred_net,
                          args=vision_features,
                          f=f"{output_model_path}/distance.onnx",
                          opset_version=17,
                          input_names=['vision_features'])

        noisy_action = torch.randn((num_samples, len_traj_pred, 2))
        timestep = torch.tensor(0)

        noise_predictor = NoisePredictor(config)
        noise_predictor.noise_pred_net = model.noise_pred_net

        # Action diffuser
        torch.onnx.export(model=noise_predictor,
                          args=(noisy_action, timestep, vision_features),
                          f=f"{output_model_path}/action.onnx",
                          opset_version=17,
                          input_names=['sample', 'timestep', 'global_cond'])

    else:
        torch.onnx.export(model=model,
                          args=tuple(input_tensor),
                          f=f"{output_model_path}",
                          opset_version=17,
                          input_names=['obs_img', 'goal_img']
                          )


class NoisePredictor(nn.Module):
    """
    Used for converting torch model to onnx.
    """

    def __init__(self, config):
        super().__init__()

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=config["encoding_size"],
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        )

    def forward(self, sample, timestep, global_condition):
        noise_pred = self.noise_pred_net(
            sample=sample,
            timestep=timestep,
            global_cond=global_condition
        )
        return noise_pred
