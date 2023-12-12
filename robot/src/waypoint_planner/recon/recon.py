import math
import random
from typing import TypeVar

import PIL
import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningDataModule
from torch import optim
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from tqdm import tqdm

from model.vae import WaypointVAEGNM
from recon.util import bytes2im, latlong_to_utm, get_files_ending_with
from recon.recon_ignore import IGNORE_LIST

Tensor = TypeVar('torch.tensor')

# Take first 90% of tracks for training, leave rest for validation
TRAIN_RATIO = 0.9

class ReconDataset(Dataset):
    MIN_WAYPOINT_DISTANCE = 5

    def __init__(self, hdf5_file_path, transform=None, **kwargs):
        self.hdf5_file = h5py.File(hdf5_file_path, 'r')
        self.transform = transform
        self.data_len = len(self.hdf5_file['imu/compass_bearing'])

        self.waypoint_spacing = kwargs["waypoint_spacing"]
        self.pred_trajectory_length = kwargs["len_traj_pred"]

        # length + 1 as we need to include current position in temporal length calculation
        self.trajectory_temporal_length = (self.pred_trajectory_length + 1) * self.waypoint_spacing

    def __len__(self):
        return max(self.data_len - self.trajectory_temporal_length, 0)

    def __getitem__(self, idx):
        # Current data
        vehicle_latitude, vehicle_longitude = self.hdf5_file['gps/latlong'][idx]
        vehicle_bearing_rad = self.hdf5_file['imu/compass_bearing'][idx]
        current_image = PIL.Image.fromarray(bytes2im(self.hdf5_file['images/rgb_left'][idx]))

        if self.transform:
            current_image = self.transform(current_image).float()

        # Calculate trajectory
        waypoint_ids = np.arange(self.waypoint_spacing, self.trajectory_temporal_length, self.waypoint_spacing) + idx
        trajectory = []
        for wp_id in waypoint_ids:
            waypoint_latitude, waypoint_longitude = self.hdf5_file['gps/latlong'][wp_id]
            wp_local_x, wp_local_y = self.calculate_local_pos(vehicle_latitude, vehicle_longitude, vehicle_bearing_rad,
                                                              waypoint_latitude, waypoint_longitude)
            trajectory.append((wp_local_x, wp_local_y))

        action_labels = torch.tensor(trajectory).float()

        # Sample a waypoint
        waypoint_idx = random.randint(idx + self.MIN_WAYPOINT_DISTANCE, self.data_len - 1)  # TODO: MAX_DISTANCE???
        waypoint_image = PIL.Image.fromarray(bytes2im(self.hdf5_file['images/rgb_left'][waypoint_idx]))

        if self.transform:
            waypoint_image = self.transform(waypoint_image).float()

        distance_label = torch.tensor((waypoint_idx - idx) / 15.0).float()  # TODO: remove magic value

        return current_image, waypoint_image, (action_labels, distance_label)

    @staticmethod
    def calculate_local_pos(vehicle_latitude, vehicle_longitude, vehicle_bearing_rad,
                            waypoint_latitude, waypoint_longitude):

        x_vehicle, y_vehicle = latlong_to_utm((vehicle_latitude, vehicle_longitude))
        x_waypoint, y_waypoint = latlong_to_utm((waypoint_latitude, waypoint_longitude))
        relative_x = x_waypoint - x_vehicle
        relative_y = y_waypoint - y_vehicle
        local_x = relative_x * math.cos(vehicle_bearing_rad) + relative_y * math.sin(vehicle_bearing_rad)
        local_y = -relative_x * math.sin(vehicle_bearing_rad) + relative_y * math.cos(vehicle_bearing_rad)
        return local_x, local_y

class ReconDataModule(LightningDataModule):

    def __init__(self, dataset_root, data_config):
        super().__init__()
        self.data_config = data_config
        self.image_transform = transforms.Compose([transforms.CenterCrop((270, 480)),
                                                   transforms.Resize(size=(90, 160)),
                                                   transforms.ToTensor()])
        self.hdf5_fnames = get_files_ending_with(dataset_root, '.hdf5')
        self.train_length = int(TRAIN_RATIO * len(self.hdf5_fnames))

    def train_dataloader(self):
        devel_datasets = []
        for f in self.hdf5_fnames[0:self.train_length]:
            if f not in IGNORE_LIST:  # TODO: move to init
                dataset = ReconDataset(f, transform=self.image_transform, **self.data_config)
                if len(dataset) > 5:
                    devel_datasets.append(dataset)

        return torch.utils.data.DataLoader(ConcatDataset(devel_datasets), num_workers=16, batch_size=64, shuffle=True, drop_last=True)

    def val_dataloader(self):
        devel_datasets = []
        for f in self.hdf5_fnames[self.train_length:]:
            if f not in IGNORE_LIST:  # TODO: move to init
                dataset = ReconDataset(f, transform=self.image_transform, **self.data_config)
                if len(dataset) > 5:
                    devel_datasets.append(dataset)

        return torch.utils.data.DataLoader(ConcatDataset(devel_datasets), num_workers=16, batch_size=64, shuffle=False, drop_last=True)

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()


class ReconExperiment(pl.LightningModule):

    def __init__(self, vae_model: WaypointVAEGNM, params: dict) -> None:
        super(ReconExperiment, self).__init__()
        self.model = vae_model
        self.params = params

    def forward(self, obs_img: torch.tensor, goal_img: torch.tensor, **kwargs) -> Tensor:
        return self.model(obs_img, goal_img, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        obs_img, goal_img, labels = batch
        results = self.forward(obs_img, goal_img)
        train_loss = self.model.loss_function(*results,
                                              labels=labels,
                                              M_N=self.params['kld_weight'],
                                              distance_loss_weight=self.params['distance_loss_weight'],
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.log("loss", train_loss['loss'], prog_bar=True, sync_dist=True)
        self.log_dict({key: val.item() for key, val in train_loss.items() if key != 'loss'}, sync_dist=True)
        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        obs_img, goal_img, labels = batch
        results = self.forward(obs_img, goal_img)
        val_loss = self.model.loss_function(*results,
                                            labels=labels,
                                            M_N=1.0,
                                            distance_loss_weight=self.params['distance_loss_weight'],
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        self.log("val_loss", val_loss['loss'], prog_bar=True, sync_dist=True)
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items() if key != 'loss'}, sync_dist=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        obs_img, goal_img, labels = batch
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


def validate_dataset(dataset_root):
    failed_files = []

    hdf5_fnames = tqdm(get_files_ending_with(dataset_root, '.hdf5'))
    for f in hdf5_fnames:
        if f not in IGNORE_LIST:  # TODO: move to init
            dataset = ReconDataset(f)
            if len(dataset) > 5:
                desc = f.split("/")[-1] + " | " + str(len(failed_files))
                hdf5_fnames.set_description(desc)
                try:
                    for i in range(len(dataset)):
                        _, _, labels = dataset[i]
                        if(labels.mean()) > 100.0:
                            failed_files.append(f)
                except:
                    failed_files.append(f)
    return failed_files
