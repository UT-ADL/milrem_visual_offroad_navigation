
from typing import Optional, Tuple
from typing import TypeVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from gnm_train.data.gnm_dataset import DATALOADER_MAPPING
from gnm_train.models.base_model import BaseModel
from gnm_train.models.modified_mobilenetv2 import MobileNetEncoder
from model.experiment import BaseExperiment

Tensor = TypeVar('torch.tensor')


class GNM(BaseModel):
    """
    Adapted from https://github.com/PrieureDeSion/drive-any-robot.
    Architecture and used configuration should be kept unchanged, otherwise pretrained weights can't be used.
    """

    def __init__(
        self,
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
        super(GNM, self).__init__(context_size, len_traj_pred, learn_angle)
        mobilenet = MobileNetEncoder(num_images=1 + self.context_size)
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
            nn.Linear(32, 1),
        )
        self.action_predictor = nn.Sequential(
            nn.Linear(32, self.len_trajectory_pred * self.num_action_params),
        )

    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        action_pred = self.action_predictor(z)

        # augment outputs to match labels size-wise
        action_pred = action_pred.reshape(
            (action_pred.shape[0], self.len_trajectory_pred, self.num_action_params)
        )
        action_pred[:, :, :2] = torch.cumsum(
            action_pred[:, :, :2], dim=1
        )  # convert position deltas into waypoints
        if self.learn_angle:
            action_pred[:, :, 2:] = F.normalize(
                action_pred[:, :, 2:].clone(), dim=-1
            )  # normalize the angle prediction
        return dist_pred, action_pred


class GNMExperiment(BaseExperiment):

    def __init__(self, model, alpha: float, learning_rate: float,
                 num_images_log: int, normalize: bool, project_folder: str) -> None:
        super(GNMExperiment, self).__init__(project_folder, num_images_log, normalize)
        self.model = model
        self.alpha = alpha
        self.learning_rate = learning_rate

    def forward(self, obs_img: torch.tensor, goal_img: torch.tensor, **kwargs) -> Tensor:
        return self.model(obs_img, goal_img, **kwargs)

    def calculate_losses(self, distance_pred, distance_labels, action_pred, action_labels):
        dist_loss = F.mse_loss(distance_pred, distance_labels)
        action_loss = F.mse_loss(action_pred, action_labels)

        total_loss = self.alpha * (1e-3 * dist_loss) + (1 - self.alpha) * action_loss

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
            'distance_loss': dist_loss.detach(),
            'action_waypts_cos_sim': action_waypts_cos_sim,
            'multi_action_waypts_cos_sim': multi_action_waypts_cos_sim
        }

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        (dist_obs_image, dist_goal_image, dist_trans_obs_image,
         dist_trans_goal_image, distance_labels, dist_dataset_index) = batch["distance"]
        distance_pred, _ = self.forward(dist_trans_obs_image, dist_trans_goal_image)

        (action_obs_image, action_goal_image, action_trans_obs_image, action_trans_goal_image,
         action_goal_pos, action_labels, action_dataset_index) = batch["action"]
        _, action_pred = self.forward(action_trans_obs_image, action_trans_goal_image)

        train_loss = self.calculate_losses(distance_pred, distance_labels, action_pred, action_labels)

        self.log("loss", train_loss['loss'], prog_bar=True, sync_dist=True)
        self.log_dict({key: val.item() for key, val in train_loss.items() if key != 'loss'}, sync_dist=True)

        if batch_idx == 0:
            self.log_predictions(batch, distance_pred, action_pred, "train")  # TODO: remove ugly access pattern

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, dataloader_idx):
        dataloader_name = DATALOADER_MAPPING[dataloader_idx]

        (dist_obs_image, dist_goal_image, dist_trans_obs_image,
         dist_trans_goal_image, distance_labels, dist_dataset_index) = batch["distance"]
        distance_pred, _ = self.forward(dist_trans_obs_image, dist_trans_goal_image)

        (action_obs_image, action_goal_image, action_trans_obs_image, action_trans_goal_image,
         action_goal_pos, action_labels, action_dataset_index) = batch["action"]
        _, action_pred = self.forward(action_trans_obs_image, action_trans_goal_image)

        val_loss = self.calculate_losses(distance_pred, distance_labels, action_pred, action_labels)

        self.log(f"{dataloader_name}_val_loss", val_loss['loss'], prog_bar=True, sync_dist=True, add_dataloader_idx=False)
        log_dict = {f"{dataloader_name}_val_{key}": val.item() for key, val in val_loss.items() if key != 'loss'}
        self.log_dict(log_dict, sync_dist=True, add_dataloader_idx=False)

        if batch_idx == 0:
            self.log_predictions(batch, distance_pred, action_pred, f"{dataloader_name}_val")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        obs_img, goal_img, labels, data = batch
        return self.forward(obs_img, goal_img)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.learning_rate)

        return {
            "optimizer": optimizer
        }
