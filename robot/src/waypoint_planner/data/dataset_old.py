import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_lightning import LightningDataModule
from tf.transformations import euler_from_quaternion
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms

from pytransform3d import rotations as pr
from pytransform3d import transformations as pt


class MilremDataset(Dataset):
    # TODO: move to yaml config
    FRAMES_PER_SECOND = 15.0
    # Sampled waypoint temporal range in frames
    MIN_WAYPOINT_DISTANCE = 5
    MAX_WAYPOINT_DISTANCE = 75

    # Used to filter out bad data
    POSITIONAL_TOLERANCE = 1.0
    #VELOCITY_MIN_TOLERANCE = 0.1
    #VELOCITY_MAX_TOLERANCE = 5.0

    def __init__(self, dataset_dir=None, transform=None, frame_range=None, **kwargs):
        self.dataset_dir = dataset_dir
        self.transform = transform

        self.context_length = kwargs["context_size"]
        self.waypoint_spacing = 15#kwargs["waypoint_spacing"]
        self.pred_trajectory_length = kwargs["len_traj_pred"]

        self.metadata = self.get_metadata(frame_range)

        # length + 1 as we need to include current position in temporal length calculation
        self.context_temporal_length = (self.context_length + 1) * self.waypoint_spacing
        self.trajectory_temporal_length = (self.pred_trajectory_length + 1) * self.waypoint_spacing
        self.data_len = len(self.metadata) - self.context_temporal_length - max(self.trajectory_temporal_length,
                                                                                self.MAX_WAYPOINT_DISTANCE)

    def get_metadata(self, frame_range):
        metadata = pd.read_csv(self.dataset_dir / 'csv/extracted_data.csv')

        # metadata may be unordered, we have to order it by time and re-create the index
        metadata.sort_values(by=["timestamp"], inplace=True)
        # reset index and store original indexes in the 'index' column for debugging
        metadata.reset_index(inplace=True)

        if frame_range:
            metadata = metadata.iloc[frame_range].copy()

        metadata['velocity'] = np.sqrt(metadata['gnss_velocity_x'] ** 2 +
                                       metadata['gnss_velocity_y'] ** 2 +
                                       metadata['gnss_velocity_z'] ** 2)

        filtered_data = metadata.dropna(subset=['camera_position_x', 'camera_position_y', 'camera_position_z']).copy()

        filtered_data['diff_x'] = filtered_data['camera_position_x'].diff().abs()
        filtered_data['diff_y'] = filtered_data['camera_position_y'].diff().abs()
        filtered_data['diff_z'] = filtered_data['camera_position_z'].diff().abs()

        filtered_metadata = filtered_data[(filtered_data['diff_x'] < self.POSITIONAL_TOLERANCE) &
                                     (filtered_data['diff_y'] < self.POSITIONAL_TOLERANCE) &
                                     (filtered_data['diff_z'] < self.POSITIONAL_TOLERANCE)]

        #filtered_metadata = filtered_metadata[filtered_metadata['velocity'] > self.VELOCITY_MIN_TOLERANCE]
        #filtered_metadata = filtered_metadata[filtered_metadata['velocity'] < self.VELOCITY_MAX_TOLERANCE]

        # create new index after filtering data, drop previous index
        filtered_metadata.reset_index(inplace=True, drop=True)

        return filtered_metadata

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        current_pos_loc = self.calculate_current_pos_loc(idx)
        current_pos_data = self.metadata.iloc[current_pos_loc]

        observations = self.load_observations(current_pos_loc)
        trajectory = self.load_trajectory(current_pos_loc)
        waypoint_data, waypoint_img, waypoint_distance = self.sample_waypoint(current_pos_loc)

        distance_label = torch.tensor([waypoint_distance / self.FRAMES_PER_SECOND]).float()
        action_labels = trajectory[:, 0:2]  # Only x, y coordinates are predicted, z is ignored

        metadata = {
            "idx": current_pos_data.name,
            "cur_pos_x": current_pos_data["camera_position_x"],
            "cur_pos_y": current_pos_data["camera_position_y"],
            "wp_idx": waypoint_data.name,
            "camera_position_x": current_pos_data["camera_position_x"],
            "camera_position_y": current_pos_data["camera_position_y"],
            "camera_position_z": current_pos_data["camera_position_z"],
            "camera_orientation_x": current_pos_data["camera_orientation_x"],
            "camera_orientation_y": current_pos_data["camera_orientation_y"],
            "camera_orientation_z": current_pos_data["camera_orientation_z"],
            "camera_orientation_w": current_pos_data["camera_orientation_w"],
            #"dataset_dir": self.dataset_dir
        }  # TODO: remove
        return observations, waypoint_img, (action_labels, distance_label), metadata

    def load_observations(self, current_pos_loc):
        observation_images = []
        context_ids = np.arange(0, self.context_temporal_length, self.waypoint_spacing) + current_pos_loc
        for i in context_ids:
            img_path = self.dataset_dir / "images" / self.metadata.iloc[i]['image_name']
            image = Image.open(str(img_path))

            if self.transform:
                image = self.transform(image)

            observation_images.append(image)
        observations = torch.cat(observation_images)
        return observations

    def sample_waypoint(self, current_pos_loc):
        waypoint_distance = random.randint(self.MIN_WAYPOINT_DISTANCE, self.MAX_WAYPOINT_DISTANCE)
        wp_idx = min(current_pos_loc + waypoint_distance, len(self.metadata) - 1)
        waypoint_data = self.metadata.iloc[wp_idx]
        waypoint_img = Image.open(str(self.dataset_dir / "images" / waypoint_data["image_name"]))
        if self.transform:
            waypoint_img = self.transform(waypoint_img)
        return waypoint_data, waypoint_img, waypoint_distance

    def calculate_current_pos_loc(self, idx):
        # All ids are shifted to provided long enough context in the beginning of the dataset
        return self.context_length * self.waypoint_spacing + idx

    def load_trajectory(self, current_pos_loc):
        vehicle_row = self.metadata.iloc[current_pos_loc]

        waypoint_ids = np.arange(self.waypoint_spacing, self.trajectory_temporal_length, self.waypoint_spacing) + current_pos_loc
        waypoints = []
        for wp_idx in waypoint_ids:
            waypoint_row = self.metadata.iloc[wp_idx]

            wp_local = self.transform_to_local(vehicle_row, waypoint_row)
            waypoints.append((wp_local[0], wp_local[1], wp_local[2]))
        return torch.tensor(waypoints).float()

    def transform_to_local(self, vehicle_row, waypoint_row):
        quaternion = [
            vehicle_row['camera_orientation_x'], vehicle_row['camera_orientation_y'],
            vehicle_row['camera_orientation_z'], vehicle_row['camera_orientation_w']
        ]
        roll, pitch, yaw = euler_from_quaternion(quaternion)
        rot_mat = pr.active_matrix_from_intrinsic_euler_xyz((roll, pitch, yaw))
        translate_mat = np.array(
            [vehicle_row["camera_position_x"], vehicle_row["camera_position_y"], vehicle_row["camera_position_z"]])
        wp_trans = pt.transform_from(rot_mat, translate_mat)
        wp_global = np.array([waypoint_row["camera_position_x"], waypoint_row["camera_position_y"],
                              waypoint_row["camera_position_z"], 1])
        wp_local = pt.transform(pt.invert_transform(wp_trans), wp_global)
        return wp_local


class MilremVizDataset(MilremDataset):
    """
    Adds unprocessed current observation information to result for visualisation purposes.
    """
    def __getitem__(self, idx):
        observations, sampled_waypoint, labels, data = super().__getitem__(idx)

        current_pos_idx = data["idx"]
        img_path = self.dataset_dir / "images" / self.metadata.iloc[current_pos_idx]['image_name']
        current_obs_img = Image.open(str(img_path))

        wp_pos_idx = data["wp_idx"]
        img_path = self.dataset_dir / "images" / self.metadata.iloc[wp_pos_idx]['image_name']
        waypoint_img = Image.open(str(img_path))

        return observations, sampled_waypoint, labels, data, current_obs_img, waypoint_img


class MilremDataModule(LightningDataModule):

    def __init__(self, root, config):
        super().__init__()
        self.root = root
        self.config = config
        image_size = config['image_size'][::-1]
        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(image_size, antialias=None),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def train_dataloader(self):
        devel_dataset = ConcatDataset([
            MilremDataset(self.root / '2023-04-12-16-02-01', self.image_transform, frame_range=slice(None, 39037),
                          **self.config),
            MilremDataset(self.root / '2023-04-13-16-50-11', self.image_transform, frame_range=slice(7200, 65171),
                          **self.config),
            MilremDataset(self.root / '2023-04-19-15-22-36', self.image_transform, frame_range=slice(None, 30221),
                          **self.config),
            MilremDataset(self.root / '2023-04-20-17-33-33', self.image_transform, frame_range=slice(None, 42114),
                          **self.config),
            MilremDataset(self.root / '2023-04-27-16-42-40', self.image_transform, frame_range=slice(None, 51836),
                          **self.config),
            MilremDataset(self.root / '2023-05-03-19-07-25', self.image_transform, frame_range=slice(5000, 18500),
                          **self.config),
            MilremDataset(self.root / '2023-05-03-19-07-25', self.image_transform, frame_range=slice(20000, 24900),
                          **self.config),
            MilremDataset(self.root / '2023-05-04-15-58-50', self.image_transform, frame_range=slice(9216, 36736),
                          **self.config),
            MilremDataset(self.root / '2023-05-10-15-41-04', self.image_transform, frame_range=slice(None, 13000),
                          **self.config),
            MilremDataset(self.root / '2023-05-11-17-08-21', self.image_transform, frame_range=slice(None, None),
                          **self.config),
            MilremDataset(self.root / '2023-05-17-15-30-02', self.image_transform, frame_range=slice(None, 5800),
                          **self.config),
            MilremDataset(self.root / '2023-05-18-16-40-47', self.image_transform, frame_range=slice(None, 9000),
                          **self.config),
            MilremDataset(self.root / '2023-05-18-16-57-00', self.image_transform, frame_range=slice(None, 28767),
                          **self.config),
            MilremDataset(self.root / '2023-05-23-15-40-24', self.image_transform, frame_range=slice(3000, 47000),
                          **self.config),
            MilremDataset(self.root / '2023-05-25-16-33-18', self.image_transform, frame_range=slice(6062, 42753),
                          **self.config),
            MilremDataset(self.root / '2023-05-30-15-42-35', self.image_transform, frame_range=slice(None, 50068),
                          **self.config),
            MilremDataset(self.root / '2023-06-01-18-10-55', self.image_transform, frame_range=slice(None, 33514),
                          **self.config),
            MilremDataset(self.root / '2023-06-06-15-41-21', self.image_transform, frame_range=slice(None, 8200),
                          **self.config),
            MilremDataset(self.root / '2023-06-06-15-41-21', self.image_transform, frame_range=slice(8300, 40000),
                          **self.config),
            MilremDataset(self.root / '2023-06-08-18-50-17', self.image_transform, frame_range=slice(4500, 18200),
                          **self.config),
            MilremDataset(self.root / '2023-06-13-15-14-21', self.image_transform, frame_range=slice(7000, 15000),
                          **self.config),
            MilremDataset(self.root / '2023-06-13-15-14-21', self.image_transform, frame_range=slice(15100, 24000),
                          **self.config),
            MilremDataset(self.root / '2023-06-13-15-49-17', self.image_transform, frame_range=slice(None, 5200),
                          **self.config),
            MilremDataset(self.root / '2023-06-13-15-49-17', self.image_transform, frame_range=slice(5500, 8700),
                          **self.config),
            MilremDataset(self.root / '2023-06-13-15-49-17', self.image_transform, frame_range=slice(8800, 39000),
                          **self.config),
            MilremDataset(self.root / '2023-06-30-12-11-33', self.image_transform, frame_range=slice(None, 61479),
                          **self.config),
            MilremDataset(self.root / '2023-07-04-15-04-53', self.image_transform, frame_range=slice(None, 20240),
                          **self.config),
            MilremDataset(self.root / '2023-07-06-12-20-35', self.image_transform, frame_range=slice(None, 5587),
                          **self.config),
            MilremDataset(self.root / '2023-07-06-12-20-35', self.image_transform, frame_range=slice(5588, 7400),
                          **self.config),
            MilremDataset(self.root / '2023-07-06-12-20-35', self.image_transform, frame_range=slice(7500, 8987),
                          **self.config),
            MilremDataset(self.root / '2023-07-06-12-20-35', self.image_transform, frame_range=slice(12500, 24724),
                          **self.config),
            MilremDataset(self.root / '2023-07-07-13-26-44', self.image_transform, frame_range=slice(None, 57422),
                          **self.config),
            MilremDataset(self.root / '2023-07-11-15-44-44', self.image_transform, frame_range=slice(None, 60897),
                          **self.config),
            MilremDataset(self.root / '2023-07-13-10-42-27', self.image_transform, frame_range=slice(None, 66477),
                          **self.config),
        ])

        return torch.utils.data.DataLoader(devel_dataset,
                                           num_workers=self.config["train_num_workers"],
                                           batch_size=self.config["train_batch_size"],
                                           shuffle=False,
                                           drop_last=True)

    def val_dataloader(self):
        valid_dataset = ConcatDataset([
            MilremDataset(self.root / '2023-04-12-16-02-01', self.image_transform, frame_range=slice(39038, None),
                          **self.config),
            MilremDataset(self.root / '2023-04-13-16-50-11', self.image_transform, frame_range=slice(65172, None),
                          **self.config),
            MilremDataset(self.root / '2023-04-19-15-22-36', self.image_transform, frame_range=slice(30222, None),
                          **self.config),
            MilremDataset(self.root / '2023-04-20-17-33-33', self.image_transform, frame_range=slice(42115, 45600),
                          **self.config),
            MilremDataset(self.root / '2023-04-27-16-42-40', self.image_transform, frame_range=slice(51837, 55500),
                          **self.config),
            MilremDataset(self.root / '2023-05-03-19-07-25', self.image_transform, frame_range=slice(25416, None),
                          **self.config),
            MilremDataset(self.root / '2023-05-04-15-58-50', self.image_transform, frame_range=slice(36737, None),
                          **self.config),
            MilremDataset(self.root / '2023-05-10-15-41-04', self.image_transform, frame_range=slice(14000, None),
                          **self.config),
            MilremDataset(self.root / '2023-05-18-16-40-47', self.image_transform, frame_range=slice(9001, 10800),
                          **self.config),
            MilremDataset(self.root / '2023-05-18-16-57-00', self.image_transform, frame_range=slice(28768, None),
                          **self.config),
            MilremDataset(self.root / '2023-05-23-15-40-24', self.image_transform, frame_range=slice(47001, 52000),
                          **self.config),
            MilremDataset(self.root / '2023-05-25-16-33-18', self.image_transform, frame_range=slice(42754, None),
                          **self.config),
            MilremDataset(self.root / '2023-05-30-15-42-35', self.image_transform, frame_range=slice(50069, None),
                          **self.config),
            MilremDataset(self.root / '2023-06-01-18-10-55', self.image_transform, frame_range=slice(33515, None),
                          **self.config),
            MilremDataset(self.root / '2023-06-06-15-41-21', self.image_transform, frame_range=slice(43000, None),
                          **self.config),
            MilremDataset(self.root / '2023-06-08-19-18-03', self.image_transform, frame_range=slice(500, 2800),
                          **self.config),
            MilremDataset(self.root / '2023-06-13-15-14-21', self.image_transform, frame_range=slice(24001, None),
                          **self.config),
            MilremDataset(self.root / '2023-06-13-15-49-17', self.image_transform, frame_range=slice(39001, 43000),
                          **self.config),
            MilremDataset(self.root / '2023-06-30-12-11-33', self.image_transform, frame_range=slice(61480, None),
                          **self.config),
            MilremDataset(self.root / '2023-07-04-15-04-53', self.image_transform, frame_range=slice(20241, None),
                          **self.config),
            MilremDataset(self.root / '2023-07-06-12-20-35', self.image_transform, frame_range=slice(24725, None),
                          **self.config),
            MilremDataset(self.root / '2023-07-07-13-26-44', self.image_transform, frame_range=slice(57423, None),
                          **self.config),
            MilremDataset(self.root / '2023-07-11-15-44-44', self.image_transform, frame_range=slice(60898, None),
                          **self.config),
            MilremDataset(self.root / '2023-07-13-10-42-27', self.image_transform, frame_range=slice(66478, None),
                          **self.config),
        ])

        return torch.utils.data.DataLoader(valid_dataset,
                                           num_workers=self.config["val_num_workers"],
                                           batch_size=self.config["val_num_workers"],
                                           drop_last=True)

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()
