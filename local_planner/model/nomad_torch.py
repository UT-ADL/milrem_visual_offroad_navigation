import torch
from diffusers import DDPMScheduler
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from torch import nn

from model.nomad_util import get_action
from vint_train.models.nomad.nomad import DenseNetwork
from vint_train.models.nomad.nomad_vint import NoMaD_ViNT, replace_bn_with_gn


class NoMaDDiffuser(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_samples = config["num_samples"]
        self.len_traj_pred = config["len_traj_pred"]
        self.num_diffusion_iters = config["num_diffusion_iters"]

        vision_encoder = NoMaD_ViNT(
            obs_encoding_size=config["encoding_size"],
            context_size=config["context_size"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
        self.vision_encoder = replace_bn_with_gn(vision_encoder)

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        self.noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=config["encoding_size"],
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        )

        self.dist_pred_net = DenseNetwork(embedding_dim=config["encoding_size"])

    def forward(self, obs_tensor, goal_tensor, is_goal_conditioned):
        vision_features = self.encode_features(obs_tensor, goal_tensor, is_goal_conditioned)
        predicted_actions = self.predict_actions(vision_features)
        predicted_distance = self.predict_distance(vision_features)
        return predicted_distance, predicted_actions

    def encode_features(self, obs_tensor, goal_tensor, is_goal_conditioned):
        if is_goal_conditioned:
            mask = torch.zeros(1).long().to(obs_tensor.device)
            obsgoal_cond = self.vision_encoder(obs_img=obs_tensor.repeat(goal_tensor.shape[0], 1, 1, 1),
                                               goal_img=goal_tensor,
                                               input_goal_mask=mask.repeat(goal_tensor.shape[0]))
            obs_cond = obsgoal_cond[0].unsqueeze(0)
        else:
            mask = torch.ones(1).long().to(obs_tensor.device)
            obs_cond = self.vision_encoder(obs_img=obs_tensor, goal_img=goal_tensor, input_goal_mask=mask)

        # (B, obs_horizon * obs_dim)
        if len(obs_cond.shape) == 2:
            obs_cond = obs_cond.repeat(self.num_samples, 1)
        else:
            obs_cond = obs_cond.repeat(self.num_samples, 1, 1)
        return obs_cond

    def predict_distance(self, vision_features):
        predicted_distances = self.dist_pred_net(vision_features)
        return predicted_distances[0]  # take first, in our case they should be all the same as goal image is same

    def predict_actions(self, obs_cond):
        # initialize action from Gaussian noise
        noisy_action = torch.randn((self.num_samples, self.len_traj_pred, 2), device=obs_cond.device)
        naction = noisy_action
        # init scheduler
        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        for k in self.noise_scheduler.timesteps[:]:
            # predict noise
            noise_pred = self.noise_pred_net(
                sample=naction,
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample
        predicted_actions = get_action(naction).squeeze()
        return predicted_actions
