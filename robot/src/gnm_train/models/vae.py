from typing import List, TypeVar

import torch
from torch import nn
from gnm_train.models.base_model import BaseModel
from gnm_train.models.modified_mobilenetv2 import MobileNetEncoder

Tensor = TypeVar('torch.tensor')


class WaypointVAEGNM(BaseModel):
    """
    VAE base arcitecture is adapted from https://github.com/AntixK/PyTorch-VAE
    GNM architecture is based on https://github.com/PrieureDeSion/drive-any-robot
    """
    def __init__(self, context_size, obs_encoding_size, goal_encoding_size, latent_dim, len_trajectory_pred, learn_angle) -> None:
        super(WaypointVAEGNM, self).__init__(context_size, len_trajectory_pred, learn_angle)

        obs_encoding_size = obs_encoding_size
        goal_encoding_size = goal_encoding_size
        self.latent_dim = latent_dim

        mobilenet = MobileNetEncoder(num_images=1 + self.context_size)
        self.obs_mobilenet = mobilenet.features
        self.compress_observation = nn.Sequential( # TODO: this is not same as goal compression
            nn.Linear(mobilenet.last_channel, obs_encoding_size),
            nn.ReLU(),
        )

        # stack the goal and the current observation
        stacked_mobilenet = MobileNetEncoder(num_images=2 + self.context_size)
        self.goal_mobilenet = stacked_mobilenet.features
        self.compress_goal = nn.Sequential(
            # there is additional linear layer here, no specific reason other than it was done so in original GNM repo
            nn.Linear(stacked_mobilenet.last_channel, 1024),
            nn.ReLU(),
            nn.Linear(1024, goal_encoding_size),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(goal_encoding_size, self.latent_dim)
        self.fc_var = nn.Linear(goal_encoding_size, self.latent_dim)

        self.feature_dropout = nn.Dropout(p=0.2)
        self.z_dropout = nn.Dropout1d(p=0.2)

        # TODO: so deep is probably not needed and one layer is sufficient as in GNM repo.
        #  This is also not good for overfitting.
        self.regressor = nn.Sequential(
            nn.Linear(obs_encoding_size+self.latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
        )

        self.dist_predictor = nn.Sequential(
            nn.Linear(32, 1),
        )

        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
        )

    def encode(self, obs_img: Tensor, goal_img: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        input_tensor = torch.cat([obs_img, goal_img], dim=1)

        goal_encoding = self.goal_mobilenet(input_tensor)
        goal_encoding = self.flatten(goal_encoding)
        goal_encoding = self.compress_goal(goal_encoding)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(goal_encoding)
        log_var = self.fc_var(goal_encoding)

        return [mu, log_var]

    def decode(self, z: Tensor, obs_img: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :param input: (Tensor) [N x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        obs_encoding = self.obs_mobilenet(obs_img)
        obs_encoding = self.flatten(obs_encoding)
        obs_encoding = self.compress_observation(obs_encoding)
        result = torch.cat([z, obs_encoding], dim=1)

        x = self.regressor(result)
        action_pred = self.action_predictor(x).reshape(-1, self.len_trajectory_pred, self.num_action_params)
        # augment outputs to match labels size-wise
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(
            action_pred[:, :, :2], dim=1
        )  # convert position deltas into waypoints

        distance = self.dist_predictor(x)
        return distance, action_pred

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, obs_img: torch.tensor, goal_img: torch.tensor) -> List[Tensor]:
        mu, log_var = self.encode(obs_img, goal_img)
        z = self.reparameterize(mu, log_var)
        z = self.feature_dropout(z)
        z = self.z_dropout(z)

        return [self.decode(z, obs_img), mu, log_var, z]

    def sample(self, obs_img, num_samples: int):
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(obs_img.device)
        batch = obs_img.repeat(num_samples, 1, 1, 1)
        return self.decode(z, batch)

