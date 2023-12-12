
from typing import TypeVar

import pytorch_lightning as pl
import torch
from torch import optim
from torch.nn import functional as F

Tensor = TypeVar('torch.tensor')


class GNMExperiment(pl.LightningModule):

    def __init__(self, model, params: dict) -> None:
        super(GNMExperiment, self).__init__()
        self.model = model
        self.params = params

    def forward(self, obs_img: torch.tensor, goal_img: torch.tensor, **kwargs) -> Tensor:
        return self.model(obs_img, goal_img, **kwargs)

    @staticmethod
    def calculate_loss(pred_actions, pred_distance, labels):
        action_loss = F.mse_loss(pred_actions, labels[0]).clamp(max=1000.0)  # TODO: look into removing clamp
        distance_loss = F.mse_loss(pred_distance, labels[1]).clamp(max=1000.0)  # TODO: look into removing clamp

        alpha = 0.5  # TODO: move to config
        loss = alpha * (1e-2 * distance_loss) + (1 - alpha) * action_loss
        return loss

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        obs_img, goal_img, labels, data = batch
        pred_distance, pred_actions = self.forward(obs_img, goal_img)
        pred_xy = pred_actions[:, :, :2]  # ignore angle
        train_loss = self.calculate_loss(pred_xy, pred_distance, labels)
        self.log("loss", train_loss, sync_dist=True, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        obs_img, goal_img, labels, data = batch
        pred_distance, pred_actions = self.forward(obs_img, goal_img)

        pred_xy = pred_actions[:, :, :2]  # ignore angle
        valid_loss = self.calculate_loss(pred_xy, pred_distance, labels)
        self.log("val_loss", valid_loss, prog_bar=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        obs_img, goal_img, labels, data = batch
        return self.forward(obs_img, goal_img)

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