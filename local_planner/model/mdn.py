
"""A module for a mixture density network layer

For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.
"""
import math
from typing import TypeVar

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn import MSELoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import mobilenet_v3_small

Tensor = TypeVar('torch.tensor')

ONEOVERSQRT2PI = 1.0 / math.sqrt(2 * math.pi)


class MDN(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(MDN, self).__init__()

        self.len_trajectory_pred = kwargs["len_traj_pred"]
        self.context_size = kwargs["context_size"]
        self.num_gaussians = kwargs["num_gaussians"]

        self.model = mobilenet_v3_small()
        mixtures = nn.Sequential(
            nn.Linear(in_features=576, out_features=1024, bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, 100),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(),
            nn.Linear(100, 50),
            nn.Tanh(),
            MDNLayer(50, 2*self.len_trajectory_pred, self.num_gaussians)
        )

        input_channels = (self.context_size4 + 1) * 3
        self.model.features[0][0] = nn.Conv2d(input_channels, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier = mixtures

    def forward(self, obs_img):
        return self.model(obs_img)


class MDNExperiment(pl.LightningModule):

    def __init__(self,
                 model,
                 params: dict) -> None:
        super(MDNExperiment, self).__init__()
        self.model = model
        self.params = params
        self.criterion = MSELoss()

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.model(x, **kwargs)

    def training_step(self, batch, batch_idx):
        obs_img, goal_img, labels, data = batch
        pi, sigma, mu = self.model(obs_img)
        action_labels = labels[0].reshape(labels[0].shape[0], -1)
        train_loss = mdn_loss(pi, sigma, mu, action_labels)
        self.log("train_loss", train_loss, sync_dist=True, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        obs_img, goal_img, labels, data = batch
        pi, sigma, mu = self.model(obs_img)
        action_labels = labels[0].reshape(labels[0].shape[0], -1)

        val_loss = mdn_loss(pi, sigma, mu, action_labels)
        self.log("val_loss", val_loss, sync_dist=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3, betas=(0.9, 0.999),
                                      eps=1e-08, weight_decay=1e-02, amsgrad=False)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.1, verbose=True)

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "train_loss"}


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


def gaussian_probability(sigma, mu, target):
    """Returns the probability of `target` given MoG parameters `sigma` and `mu`.

    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        target (BxI): A batch of target. B is the batch size and I is the number of
            input dimensions.

    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """
    target = target.unsqueeze(1).expand_as(sigma)
    ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma)**2) / sigma
    return torch.prod(ret, 2)


def mdn_loss(pi, sigma, mu, target):
    """Calculates the error, given the MoG parameters and the target

    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    prob = pi * gaussian_probability(sigma, mu, target)
    prob = prob + 0.00000000001
    nll = -torch.log(torch.sum(prob, dim=1))
    return torch.mean(nll)


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
