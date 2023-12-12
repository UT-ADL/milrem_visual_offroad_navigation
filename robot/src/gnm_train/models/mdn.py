
"""A module for a mixture density network layer

For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.
"""
import math
from typing import TypeVar, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .base_model import BaseModel
from .modified_mobilenetv2 import MobileNetEncoder
import torch.nn.functional as F
import torch.distributions as D


Tensor = TypeVar('torch.tensor')

ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)


class MDN(BaseModel):

    def __init__(
        self,
        num_gaussians: int = 3,
        context_size: int = 5,
        len_traj_pred: Optional[int] = 5,
        learn_angle: Optional[bool] = True,
        obs_encoding_size: Optional[int] = 1024,
        goal_encoding_size: Optional[int] = 1024,
    ) -> None:
        """
        GNM main class
        Args:
            context_size (int): how many previous observations to used for context
            len_traj_pred (int): how many waypoints to predict in the future
            observation_encoding_size (int): size of the encoding of the observation images
            goal_encoding_size (int): size of the encoding of the goal images
            learn_angle (bool): whether to predict the yaw of the robot
        """
        super(MDN, self).__init__(context_size, len_traj_pred, learn_angle)
        mobilenet = MobileNetEncoder(num_images=1 + self.context_size)
        self.num_gaussians = num_gaussians
        self.obs_mobilenet = mobilenet.features
        self.obs_encoding_size = obs_encoding_size
        self.compress_observation = nn.Sequential(
            nn.Linear(mobilenet.last_channel, self.obs_encoding_size),
            nn.ReLU(),
        )
        stacked_mobilenet = MobileNetEncoder(
            num_images=2 + self.context_size
        )  # stack the goal and the current observation
        self.goal_mobilenet = stacked_mobilenet.features
        self.goal_encoding_size = goal_encoding_size
        self.compress_goal = nn.Sequential(
            nn.Linear(stacked_mobilenet.last_channel, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.goal_encoding_size),
            nn.ReLU(),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(self.goal_encoding_size + self.obs_encoding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
        )

        self.dist_predictor = nn.Sequential(
            MDNLayer(32, 1, self.num_gaussians)
        )
        self.action_predictor = nn.Sequential(
            MDNLayer(32, 2 * self.len_trajectory_pred, self.num_gaussians)
        )

    def forward(self, obs_img: torch.tensor, goal_img: torch.tensor):
        obs_encoding = self.obs_mobilenet(obs_img)
        obs_encoding = self.flatten(obs_encoding)
        obs_encoding = self.compress_observation(obs_encoding)

        obs_goal_input = torch.cat([obs_img, goal_img], dim=1)
        goal_encoding = self.goal_mobilenet(obs_goal_input)
        goal_encoding = self.flatten(goal_encoding)
        goal_encoding = self.compress_goal(goal_encoding)

        z = torch.cat([obs_encoding, goal_encoding], dim=1)
        z = self.linear_layers(z)
        dist_pred = self.dist_predictor(z)
        action_pi, action_sigma, action_mu = self.action_predictor(z)

        action_sigma = action_sigma.reshape((action_sigma.shape[0], self.num_gaussians, self.len_trajectory_pred, -1))
        # print("action_sigma", action_sigma.shape)
        action_mu = action_mu.reshape((action_mu.shape[0], self.num_gaussians, self.len_trajectory_pred, -1))
        # print("action_mu", action_mu.shape)
        # convert position deltas into waypoints
        # action_sigma[:, :, :, :2] = torch.cumsum(action_sigma[:, :, :, :2], dim=2)
        #action_sigma = torch.cumsum(action_sigma, dim=2)
        #action_mu = torch.cumsum(action_mu, dim=2)

        # # augment outputs to match labels size-wise
        # action_pred = action_pred.reshape(
        #     (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        # )
        # action_pred[:, :, :2] = torch.cumsum(
        #     action_pred[:, :, :2], dim=1
        # )  # convert position deltas into waypoints
        # if self.learn_angle:
        #     action_pred[:, :, 2:] = F.normalize(
        #         action_pred[:, :, 2:].clone(), dim=-1
        #     )  # normalize the angle prediction
        return dist_pred, (action_pi, action_sigma, action_mu)


class MDNLayer(nn.Module):
    """A mixture density network layer

    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.

    Adapted from https://github.com/sagelywizard/pytorch-mdn/blob/master/mdn/mdn.py

    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions

    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.

    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.

    """

    def __init__(self, in_features, out_features, num_gaussians):
        super(MDNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.pi = nn.Sequential(
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=1)
        )
        self.sigma = nn.Linear(in_features, out_features * num_gaussians)
        self.mu = nn.Linear(in_features, out_features * num_gaussians)

    def forward(self, minibatch):
        pi = self.pi(minibatch)
        sigma = torch.exp(self.sigma(minibatch))
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu


def mdn_loss(pi, sigma, mu, target):
    mix = D.Categorical(pi)
    comp = D.Independent(D.Normal(mu, sigma), 1)
    gmm = D.MixtureSameFamily(mix, comp)

    return - gmm.log_prob(target).mean()


def sample(pi, sigma, mu):
    """Draw samples from a MoG.
    """
    # Choose which gaussian we'll sample from
    pis = Categorical(pi).sample().view(pi.size(0), 1, 1).repeat(1, 1, mu.shape[2])
    # Choose a random sample, one randn for batch X output dims
    # Do a (output dims)X(batch size) tensor here, so the broadcast works in
    # the next step, but we have to transpose back.
    gaussian_noise = torch.randn((sigma.shape[0], sigma.shape[2]), requires_grad=False)
    variance_samples = sigma.gather(1, pis).detach().squeeze()
    mean_samples = mu.detach().gather(1, pis).squeeze()
    return gaussian_noise * variance_samples + mean_samples
