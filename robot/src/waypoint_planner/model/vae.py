from typing import List, TypeVar, Any

import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from gnm_train.data.gnm_dataset import DATALOADER_MAPPING
from gnm_train.models.base_model import BaseModel
from gnm_train.models.modified_mobilenetv2 import MobileNetEncoder
from gnm_train.visualizing.action_utils import visualize_traj_pred
from gnm_train.visualizing.distance_utils import visualize_dist_pred
from gnm_train.visualizing.visualize_utils import to_numpy

Tensor = TypeVar('torch.tensor')


class WaypointVAEGNM(BaseModel):
    """
    VAE base architecture is adapted from https://github.com/AntixK/PyTorch-VAE
    GNM architecture is based on https://github.com/PrieureDeSion/drive-any-robot
    """
    def __init__(self, context_size, obs_encoding_size, goal_encoding_size, latent_dim, len_trajectory_pred, learn_angle) -> None:
        super(WaypointVAEGNM, self).__init__(context_size, len_trajectory_pred, learn_angle)

        obs_encoding_size = obs_encoding_size
        goal_encoding_size = goal_encoding_size
        self.latent_dim = latent_dim

        mobilenet = MobileNetEncoder(num_images=1 + self.context_size)
        self.obs_mobilenet = mobilenet.features
        self.compress_observation = nn.Sequential(
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

    def forward(self, obs_img: torch.tensor, goal_img: torch.tensor):
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


class WaypointVAEExperiment(pl.LightningModule):

    def __init__(self, vae_model: WaypointVAEGNM, alpha: float, kld_weight: float,
                 learning_rate: float, num_images_log: int,
                 normalize: bool, project_folder: str) -> None:
        super(WaypointVAEExperiment, self).__init__()
        self.model = vae_model
        self.alpha = alpha
        self.kld_weight = kld_weight
        self.learning_rate = learning_rate
        self.num_images_log = num_images_log
        self.normalize = normalize
        self.project_folder = project_folder

    def forward(self, obs_img: torch.tensor, goal_img: torch.tensor, **kwargs) -> Tensor:
        return self.model(obs_img, goal_img, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        (dist_obs_image, dist_goal_image, dist_trans_obs_image,
         dist_trans_goal_image, distance_labels, dist_dataset_index) = batch["distance"]
        distance_pred = self.forward(dist_trans_obs_image, dist_trans_goal_image)

        (action_obs_image, action_goal_image, action_trans_obs_image, action_trans_goal_image,
         action_goal_pos, action_labels, action_dataset_index) = batch["action"]
        action_pred = self.forward(action_trans_obs_image, action_trans_goal_image)

        train_loss = self.calculate_losses(distance_pred, distance_labels, action_pred, action_labels)

        self.log("loss", train_loss['loss'], prog_bar=True, sync_dist=True)
        self.log_dict({key: val.item() for key, val in train_loss.items() if key != 'loss'}, sync_dist=True)

        if batch_idx == 0:
            self.log_predictions(batch, distance_pred[0][0], action_pred[0][1], "train")  # TODO: remove ugly access pattern

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, dataloader_idx):
        dataloader_name = DATALOADER_MAPPING[dataloader_idx]

        (dist_obs_image, dist_goal_image, dist_trans_obs_image,
         dist_trans_goal_image, distance_labels, dist_dataset_index) = batch["distance"]
        distance_pred = self.forward(dist_trans_obs_image, dist_trans_goal_image)

        (action_obs_image, action_goal_image, action_trans_obs_image, action_trans_goal_image,
         action_goal_pos, action_labels, action_dataset_index) = batch["action"]
        action_pred = self.forward(action_trans_obs_image, action_trans_goal_image)

        val_loss = self.calculate_losses(distance_pred, distance_labels, action_pred, action_labels)

        self.log(f"{dataloader_name}_val_loss", val_loss['loss'], prog_bar=True, sync_dist=True, add_dataloader_idx=False)
        log_dict = {f"{dataloader_name}_val_{key}": val.item() for key, val in val_loss.items() if key != 'loss'}
        self.log_dict(log_dict, sync_dist=True, add_dataloader_idx=False)

        if batch_idx == 0:
            self.log_predictions(batch, distance_pred[0][0], action_pred[0][1], f"{dataloader_name}_val")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        (dist_obs_image, dist_goal_image, dist_trans_obs_image,
         dist_trans_goal_image, distance_labels, dist_dataset_index) = batch["distance"]
        distance_pred = self.forward(dist_trans_obs_image, dist_trans_goal_image)

        (action_obs_image, action_goal_image, action_trans_obs_image, action_trans_goal_image,
         action_goal_pos, action_labels, action_dataset_index) = batch["action"]
        action_pred = self.forward(action_trans_obs_image, action_trans_goal_image)

        return distance_pred, action_pred

    def calculate_kld_loss(self, mu, log_var):
        dist_kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        return dist_kld_loss

    def calculate_losses(self, distance_pred, distance_labels, action_pred, action_labels):
        (dist_pred, _), dist_mu, dist_log_var, _ = distance_pred
        (_, action_pred), action_mu, action_log_var, _ = action_pred

        dist_loss = F.mse_loss(dist_pred, distance_labels)
        dist_kld_loss = self.calculate_kld_loss(dist_mu, dist_log_var)

        action_loss = F.mse_loss(action_pred, action_labels)
        action_kld_loss = self.calculate_kld_loss(action_mu, action_log_var)

        kld_loss = (action_kld_loss + dist_kld_loss) / 2.0

        recons_loss = self.alpha * (1e-3 * dist_loss) + (1 - self.alpha) * action_loss
        total_loss = recons_loss + self.kld_weight * kld_loss

        action_waypts_cos_sim = F.cosine_similarity(
            action_pred[:, :, :2], action_labels[:, :, :2], dim=-1
        ).mean()
        multi_action_waypts_cos_sim = F.cosine_similarity(
            torch.flatten(action_pred[:, :, :2], start_dim=1),
            torch.flatten(action_labels[:, :, :2], start_dim=1),
            dim=-1,
        ).mean()

        return {
            'loss': total_loss,
            'action_loss': action_loss.detach(),
            'action_kld_loss': -action_kld_loss.detach(),
            'distance_loss': dist_loss.detach(),
            'distance_kld_loss': -dist_kld_loss.detach(),
            'reconstruction_Loss': recons_loss.detach(),
            'kld_loss': -kld_loss.detach(),
            'action_waypts_cos_sim': action_waypts_cos_sim,
            'multi_action_waypts_cos_sim': multi_action_waypts_cos_sim
        }

    def log_predictions(self, batch, dist_pred, action_pred, eval_type):

        (dist_obs_image, dist_goal_image, dist_trans_obs_image,
         dist_trans_goal_image, dist_label, dist_dataset_index) = batch["distance"]

        (action_obs_image, action_goal_image, action_trans_obs_image, action_trans_goal_image,
         action_goal_pos, action_label, action_dataset_index) = batch["action"]

        visualize_dist_pred(
            to_numpy(dist_obs_image),
            to_numpy(dist_goal_image),
            to_numpy(dist_pred),
            to_numpy(dist_label),
            eval_type,
            self.project_folder,
            self.current_epoch,
            self.num_images_log,
            use_wandb=True,
            display=False
        )
        visualize_traj_pred(
            to_numpy(action_obs_image),
            to_numpy(action_goal_image),
            to_numpy(action_dataset_index),
            to_numpy(action_goal_pos),
            to_numpy(action_pred),
            to_numpy(action_label),
            eval_type,
            self.normalize,
            self.project_folder,
            self.current_epoch,
            self.num_images_log,
            use_wandb=True,
            display=False
        )

    def configure_optimizers(self):

        # TODO: weight decay?
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.learning_rate)

        # TODO: scheduler
        #scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.params['scheduler_gamma'])

        return {
            "optimizer": optimizer,
            #"lr_scheduler": scheduler
        }
