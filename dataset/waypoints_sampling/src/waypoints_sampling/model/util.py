import torch

import waypoints_sampling.gnm_train.utils as gnm_utils
from waypoints_sampling.gnm_train.models.gnm import GNM
from waypoints_sampling.model.gnm import GNMExperiment
from waypoints_sampling.model.mdn import MDN, MDNExperiment
from waypoints_sampling.model.vae import WaypointVAEGNM, WaypointVAEExperiment


def load_model(model_path, model_type, model_config):
    if model_type == "vae":
        model = load_vae_model(model_path, model_config)
    elif model_type == "mdn":
        model = load_mdn_model(model_path, model_config)
    elif model_type == "gnm":
        model = load_gnm_model(model_path, model_config)
    elif model_type == "gnm-pretrained":
        model = load_gnm_pretrained_model(model_path, model_config)
    else:
        raise Exception(f"Unknown model type {model_type}")

    return model


def load_gnm_pretrained_model(model_path, config):
    model_params = config["model_params"]
    gnm_model = gnm_utils.load_model(
        model_path,
        model_params["model_type"],
        model_params["context_length"],
        model_params["pred_trajectory_length"],
        model_params["learn_angle"],
        model_params["obs_encoding_size"],
        model_params["goal_encoding_size"],
        model_params["obsgoal_encoding_size"]
    )
    gnm_model.eval()

    return gnm_model


def load_gnm_model(model_path, config):
    model_params = config["model_params"]
    model = GNM(
                model_params["context_length"],
                model_params["pred_trajectory_length"],
                model_params["learn_angle"],
                model_params["obs_encoding_size"],
                model_params["goal_encoding_size"]
            )
    ckpt = torch.load(model_path)
    experiment = GNMExperiment(model, config['exp_params'])
    experiment.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


def load_vae_model(model_path, config):
    model = WaypointVAEGNM(**config['model_params'])
    ckpt = torch.load(model_path)
    experiment = WaypointVAEExperiment(model, config['exp_params'])
    experiment.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model


def load_mdn_model(model_path, config):
    model = MDN(**config['model_params'])
    ckpt = torch.load(model_path)
    experiment = MDNExperiment(model, config['exp_params'])
    experiment.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model
