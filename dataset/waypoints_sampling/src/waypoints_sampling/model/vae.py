from typing import List, TypeVar

import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from waypoints_sampling.gnm_train.models.modified_mobilenetv2 import MobileNetEncoder

Tensor = TypeVar('torch.tensor')


class WaypointVAEGNM(nn.Module):
    """
    VAE base arcitecture is adapted from https://github.com/AntixK/PyTorch-VAE
    GNM architecture is based on https://github.com/PrieureDeSion/drive-any-robot
    """
    
    def __init__(self, **kwargs) -> None:
        super(WaypointVAEGNM, self).__init__()

        self.context_length = kwargs["context_length"]
        obs_encoding_size = kwargs["obs_encoding_size"]
        goal_encoding_size = kwargs["goal_encoding_size"]
        self.latent_dim = kwargs["latent_dim"]
        self.len_trajectory_pred = kwargs["pred_trajectory_length"]
        self.num_samples = kwargs["num_samples"]

        self.num_action_params = 2  # Only x, y is predicted

        mobilenet = MobileNetEncoder(num_images=1 + self.context_length)
        self.obs_mobilenet = mobilenet.features
        self.compress_observation = nn.Sequential(
            nn.Linear(mobilenet.last_channel, obs_encoding_size),
            nn.ReLU(),
        )

        # stack the goal and the current observation
        stacked_mobilenet = MobileNetEncoder(num_images=2 + self.context_length)
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

    def flatten(self, z: torch.Tensor) -> torch.Tensor:
        z = F.adaptive_avg_pool2d(z, (1, 1))
        z = torch.flatten(z, 1)
        return z

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
        actions = self.action_predictor(x).reshape(-1, self.len_trajectory_pred, self.num_action_params)
        distance = self.dist_predictor(x)
        return distance, actions

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
        return [self.decode(z, obs_img), self.sample(obs_img, self.num_samples), mu, log_var, z]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        # TODO: fix confusing order
        pred_distance, pred_actions = args[0]
        mu = args[1]
        log_var = args[2]
        labels = kwargs['labels']

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        action_loss = F.mse_loss(pred_actions, labels[0]).clamp(max=1000.0)  # TODO: look into removing clamp
        distance_loss = F.mse_loss(pred_distance, labels[1]).clamp(max=1000.0)

        alpha = kwargs["distance_loss_weight"]
        recons_loss = alpha * (1e-2 * distance_loss) + (1 - alpha) * action_loss

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {
            'loss': loss,
            'action_loss': action_loss.detach(),
            'distance_loss': distance_loss.detach(),
            'reconstruction_Loss': recons_loss.detach(),
            'kld_loss': -kld_loss.detach()
        }

    def sample(self, obs_img, num_samples: int):
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(obs_img.device)
        batch = obs_img.repeat(num_samples, 1, 1, 1)
        return self.decode(z, batch)


class WaypointVAEExperiment(pl.LightningModule):

    def __init__(self, vae_model: WaypointVAEGNM, params: dict) -> None:
        super(WaypointVAEExperiment, self).__init__()
        self.model = vae_model
        self.params = params

    def forward(self, obs_img: torch.tensor, goal_img: torch.tensor, **kwargs) -> Tensor:
        return self.model(obs_img, goal_img, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        obs_img, goal_img, labels, data = batch
        results = self.forward(obs_img, goal_img)
        train_loss = self.model.loss_function(*results,
                                              labels=labels,
                                              M_N=self.params['kld_weight'],
                                              distance_loss_weight=self.params['distance_loss_weight'],
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.log("loss", train_loss['loss'], prog_bar=True, sync_dist=True)
        self.log_dict({key: val.item() for key, val in train_loss.items() if key is not 'loss'}, sync_dist=True)
        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        obs_img, goal_img, labels, data = batch
        results = self.forward(obs_img, goal_img)
        val_loss = self.model.loss_function(*results,
                                            labels=labels,
                                            M_N=1.0,
                                            distance_loss_weight=self.params['distance_loss_weight'],
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        self.log("val_loss", val_loss['loss'], prog_bar=True, sync_dist=True)
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items() if key is not 'loss'}, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        obs_img, goal_img, labels, data = batch
        return self.forward(obs_img, goal_img, labels=labels)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])

        scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                     gamma=self.params['scheduler_gamma'])

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

