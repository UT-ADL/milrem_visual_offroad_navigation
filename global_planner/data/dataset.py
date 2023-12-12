from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms

from data.mapping import MapReader
from data.mask import Mask, limit_trajectory

VELOCITY_MIN_TOLERANCE = 0.05
VELOCITY_MAX_TOLERANCE = 2.5


class TrajectoryDataset(Dataset):
    """
    Generic dataset that reads maps and trajectories from dataset.
    """
    def __init__(self, dataset_path, map_reader, map_size, trajectory_min_len, trajectory_max_len,
                 trajectory_sampling_rate, norm_trajectory=True):

        self.trajectory_min_len = trajectory_min_len
        self.trajectory_max_len = trajectory_max_len
        self.map_size = map_size
        self.trajectory_sampling_rate = trajectory_sampling_rate
        self.norm_trajectory = norm_trajectory

        metadata = get_metadata(dataset_path, sampling_rate=self.trajectory_sampling_rate)

        lat = metadata['gnss_latitude'].to_numpy()
        lon = metadata['gnss_longitude'].to_numpy()
        self.map_reader = map_reader
        trajectory_in_pix = self.map_reader.lat_lon_to_pixel(lat, lon)
        self.positions = torch.tensor([[x, y] for (x, y) in trajectory_in_pix], dtype=torch.int)

        if len(self.positions) < self.trajectory_max_len:
            print(f"Max trajectory length > than trajectory positons. Decreasing max trajectory length to {len(self.positions)}")
            self.trajectory_max_len = len(self.positions)
            self.trajectory_min_len = min(self.trajectory_min_len, self.trajectory_max_len)

    def __len__(self):
        return len(self.positions) - self.trajectory_max_len

    def __getitem__(self, idx):
        traj_length = np.random.randint(self.trajectory_min_len, self.trajectory_max_len + 1)
        traj_start = np.random.randint(0, self.trajectory_max_len + 1 - traj_length)
        start_idx = idx + traj_start

        x = self.positions[start_idx:start_idx + traj_length, 0]
        y = self.positions[start_idx:start_idx + traj_length, 1]

        center = self.positions[start_idx]
        center_x, center_y = int(center[0]), int(center[1])

        x_start = center_x - self.map_size
        y_start = center_y - self.map_size

        x_local = x - x_start
        y_local = y - y_start

        if self.norm_trajectory:
            x_local = (x_local - x_local[0]) / self.map_size
            y_local = (y_local - y_local[0]) / self.map_size

        trajectory = np.column_stack((x_local, y_local))
        trajectory_tensor = torch.tensor(trajectory, dtype=torch.float32)

        map_img = self.crop_map(start_idx)
        map_transform = create_image_transform()
        map_tensor = map_transform(map_img)

        return trajectory_tensor, map_tensor, map_img

    def crop_map(self, trajectory_start_index):
        trajectory_position = self.positions[trajectory_start_index]
        return self.map_reader.crop_map_by_position(trajectory_position)


def pad_trajectory(trajectory, max_size):
    """
    Pad the given tensor with its last value to the max_size.

    Parameters:
    - trajectory: The input tensor of shape (N, 2) where N can be <= max_size.
    - max_size: The desired length of the output tensor.

    Returns:
    - A tensor of shape (max_size, 2).
    """
    current_size = trajectory.shape[0]
    if current_size == max_size:
        return trajectory

    padding_size = max_size - current_size
    padding = trajectory[-1].repeat(padding_size, 1)  # Repeat the last point
    return torch.cat((trajectory, padding), dim=0)


class OneHotEncodedDataset(TrajectoryDataset):
    """
    Dataset used for training global planner with segmentive approach.
    """
    def __init__(self, dataset_path, map_reader, map_size,
                 trajectory_min_len, trajectory_max_len, trajectory_sampling_rate, config):
        super().__init__(dataset_path, map_reader, map_size,
                         trajectory_min_len, trajectory_max_len, trajectory_sampling_rate, norm_trajectory=False)
        self.mask = Mask(distance_threshold=config["distance_threshold"],
                         mask_type=config["mask_type"],
                         kernel_size=config["kernel_size"],
                         sigma=config["sigma"])

    def __getitem__(self, idx):
        trajectory, map_tensor, map_img = super().__getitem__(idx)
        trajectory = limit_trajectory(trajectory, map_tensor.shape[1:])
        trajectory_mask = self.mask.create_mask(map_tensor.shape[1:], trajectory)
        masked_map = self.mask.create_masked_map(map_tensor, trajectory)
        padded_trajectory = pad_trajectory(trajectory, self.trajectory_max_len)
        return masked_map, trajectory_mask, padded_trajectory, map_img


class NegativeSamplingDataset(TrajectoryDataset):
    """
    Dataset used for training global planner with contrastive approach.
    """
    def __init__(self, dataset_path, map_reader, map_size, trajectory_min_len, trajectory_max_len, trajectory_sampling_rate,
                 n_negative_samples, negative_dist_thres):
        super().__init__(dataset_path, map_reader, map_size, trajectory_min_len, trajectory_max_len, trajectory_sampling_rate)
        self.n_negative_samples = n_negative_samples
        self.negative_dist_thres = negative_dist_thres

    def __getitem__(self, idx):
        trajectory, map_tensor, map_img = super().__getitem__(idx)

        start_waypoint = trajectory[0, :]
        end_waypoint = trajectory[-1, :]

        positive_waypoints = trajectory[1:-1]
        repeated_start_waypoint = start_waypoint.unsqueeze(0).repeat(len(positive_waypoints), 1)
        repeated_end_waypoint = end_waypoint.unsqueeze(0).repeat(len(positive_waypoints), 1)
        positive_coordinates = torch.concat([repeated_start_waypoint, repeated_end_waypoint, positive_waypoints], dim=1)

        t = torch.linspace(-1.0, 1.0, 2*self.map_size)
        x, y = torch.meshgrid(t, t, indexing='ij')
        wps = torch.stack((x, y), dim=-1)

        points_expanded = wps.view(-1, 2).unsqueeze(1)
        trajectory_expanded = trajectory.unsqueeze(0)
        distances = torch.norm(points_expanded - trajectory_expanded, dim=2)
        min_distances = distances.min(dim=1).values.numpy()
        min_dist_filter = np.where(min_distances > self.negative_dist_thres)
        weights = 1 / min_distances[min_dist_filter] ** 2
        probs = weights / weights.sum()
        min_dist_filter = np.random.choice(min_dist_filter[0], self.n_negative_samples, p=probs)

        negative_waypoints = wps.view(-1, 2)
        negative_waypoints = negative_waypoints[min_dist_filter]

        repeated_start_waypoint = start_waypoint.unsqueeze(0).repeat(self.n_negative_samples, 1)
        repeated_end_waypoint = end_waypoint.unsqueeze(0).repeat(self.n_negative_samples, 1)
        negative_coordinates = torch.concat([repeated_start_waypoint, repeated_end_waypoint, negative_waypoints], dim=1)

        return positive_coordinates, negative_coordinates, map_tensor


def create_image_transform():
    transform = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    return transforms.Compose(transform)


def train_maps(map_type):
    return {
        "2023-04-12-16-02-01": f"Ujula_{map_type}.tif",
        #"2023-04-13-16-50-11": f"Kaiu_{map_type}.tif", # no trajectory
        "2023-04-19-15-22-36": f"Supilinn_{map_type}.tif",
        "2023-04-20-17-33-33": f"Apteekri_{map_type}.tif",
        "2023-04-27-16-42-40": f"Päidlapalu_{map_type}.tif",
        "2023-05-03-19-07-25": f"Vasula_{map_type}.tif",
        "2023-05-10-15-41-04": f"Annekanal_{map_type}.tif",
        "2023-05-17-15-30-02": f"Majoraadi_{map_type}.tif",
        "2023-05-18-16-40-47": f"Tehvandi_{map_type}.tif",
        "2023-05-18-16-57-00": f"Tehvandi_{map_type}.tif",
        "2023-05-23-15-40-24": f"Lajavangu_{map_type}.tif",
        "2023-05-25-16-33-18": f"Kirikuküla_{map_type}.tif",
        "2023-05-30-15-42-35": f"Voorepalu_{map_type}.tif",
        "2023-06-01-18-10-55": f"Raadi_{map_type}.tif",
        "2023-06-06-15-41-21": f"Karaski_{map_type}.tif",
        "2023-06-13-15-49-17": f"Piigaste_{map_type}.tif",
        #"2023-06-15-18-10-18": f"Supilinn_{map_type}.tif",  # 0-125 ok, otherwise problems with trajectory
        "2023-06-30-12-11-33": f"Delta_{map_type}.tif",
        "2023-07-06-12-20-35": f"Botaed_{map_type}.tif",
        "2023-07-07-13-26-44": f"Botaed_{map_type}.tif",
        "2023-07-13-10-42-27": f"Matkekeskus_{map_type}.tif",
        "2023-07-17-13-37-10": f"Raadi2_{map_type}.tif",
        "2023-07-17-14-38-28": f"Raadi2_{map_type}.tif",
        "2023-07-19-13-12-11": f"Lumepark_{map_type}.tif",
        "2023-07-21-11-52-18": f"Supilinn_{map_type}.tif",
        "2023-07-24-13-53-29": f"Annekanal_{map_type}.tif",
        "2023-07-24-14-29-06": f"Annekanal_{map_type}.tif",
        "2023-07-26-14-22-18": f"Tähtvere_{map_type}.tif",
        "2023-07-27-14-58-24": f"Matkekeskus_{map_type}.tif",
        "2023-08-01-15-47-18": f"Koigera_{map_type}.tif",
        "2023-08-02-16-27-51": f"Lodjakoda_{map_type}.tif",
        "2023-08-04-11-15-22": f"Tartu_kesklinn_{map_type}.tif",
        "2023-08-08-15-40-29": f"Hatiku_{map_type}.tif",
        "2023-08-09-13-44-25": f"Tartu_kesklinn_{map_type}.tif",
        "2023-08-09-14-07-47": f"Tartu_kesklinn_{map_type}.tif",
        "2023-08-10-16-19-31": f"Otepää_linn_{map_type}.tif",
        "2023-08-17-16-16-29": f"Purtsi_{map_type}.tif",
        "2023-08-21-15-17-51": f"Ilmatsalu_{map_type}.tif",
        "2023-08-22-15-25-30": f"Janokjärve_{map_type}.tif",
        "2023-08-23-15-04-12": f"Ihaste_{map_type}.tif",
        "2023-08-23-15-12-38": f"Ihaste_{map_type}.tif",
        "2023-08-23-15-17-22": f"Ihaste_{map_type}.tif",
        "2023-08-23-15-21-21": f"Ihaste_{map_type}.tif",
        "2023-08-24-16-09-18": f"Uniküla_{map_type}.tif",
        "2023-08-29-15-47-52": f"Porgandi_{map_type}.tif",
        "2023-08-30-17-30-45": f"Delta_{map_type}.tif",
        "2023-08-31-11-13-18": f"Lähte_{map_type}.tif",
        "2023-08-31-11-29-52": f"Lähte_{map_type}.tif",
        "2023-08-31-11-54-39": f"Lähte_{map_type}.tif",
        "2023-08-31-12-25-01": f"Lähte_{map_type}.tif",
        "2023-08-31-16-28-53": f"Välgi_{map_type}.tif",
        "2023-08-31-16-49-52": f"Välgi_{map_type}.tif",
        "2023-08-31-17-10-17": f"Välgi_{map_type}.tif",
        "2023-08-31-17-24-54": f"Välgi_{map_type}.tif",
        "2023-09-01-15-23-41": f"Tehnikamaja_{map_type}.tif",
        "2023-09-01-15-46-43": f"Tehnikamaja_{map_type}.tif",
        "2023-10-06-13-08-18": f"Hiinalinn_{map_type}.tif"
    }


def val_maps(map_type):
    return {
        "2023-05-11-17-08-21": f"Kärgandi_{map_type}.tif",
        "2023-05-11-17-54-42": f"Kärgandi_{map_type}.tif",
        "2023-05-11-18-21-37": f"Kärgandi_{map_type}.tif",
        "2023-06-08-18-50-17": f"Annelinn2_{map_type}.tif",
        "2023-06-13-15-14-21": f"Piigaste_{map_type}.tif",  # some problems with trajectory, ok for val
        "2023-07-04-15-04-53": f"Botaed_{map_type}.tif",  # Trajectory problem in the beginning, ok for val
        "2023-07-11-15-44-44": f"Palojärve_{map_type}.tif",
        "2023-07-27-15-46-09": f"Delta_{map_type}.tif",
        "2023-07-27-16-12-51": f"Delta_{map_type}.tif",
        "2023-08-08-16-37-28": f"Hatiku_{map_type}.tif",
        "2023-08-23-15-26-38": f"Ihaste_{map_type}.tif",
        "2023-08-23-15-57-55": f"Ihaste_{map_type}.tif",
        "2023-08-25-15-48-47": f"Ihaste_{map_type}.tif",
    }


def tracks_mapping(map_type):
    return train_maps(map_type) | val_maps(map_type)


class OneHotEncodedDataModule(LightningDataModule):

    def __init__(self, map_size, map_type, map_path, dataset_path, trajectory_min_len, trajectory_max_len,
                 trajectory_sampling_rate, config, batch_size=1, num_workers=0):
        super().__init__()
        self.map_size = map_size
        self.map_type = map_type
        self.map_path = Path(map_path)
        self.dataset_path = dataset_path
        self.trajectory_min_len = trajectory_min_len
        self.trajectory_max_len = trajectory_max_len
        self.trajectory_sampling_rate = trajectory_sampling_rate
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.config = config

    def train_dataloader(self):
        datasets = []
        for dataset_name, map_name in train_maps(self.map_type).items():
            print(f"Adding {dataset_name} to train data set.")
            map_reader = MapReader(self.map_path / map_name, self.map_size)
            datasets.append(OneHotEncodedDataset(dataset_path=Path(self.dataset_path) / dataset_name,
                                                 map_reader=map_reader,
                                                 map_size=self.map_size,
                                                 trajectory_min_len=self.trajectory_min_len,
                                                 trajectory_max_len=self.trajectory_max_len,
                                                 trajectory_sampling_rate=self.trajectory_sampling_rate,
                                                 config=self.config))
        dataloader = DataLoader(ConcatDataset(datasets), batch_size=self.batch_size, shuffle=True,
                                pin_memory=True, num_workers=self.num_workers, drop_last=True)
        print(f"Train dataset length {len(dataloader)}")
        return dataloader

    def val_dataloader(self):
        datasets = []
        for dataset_name, map_name in val_maps(self.map_type).items():
            print(f"Adding {dataset_name} to val data set.")
            map_reader = MapReader(self.map_path / map_name, self.map_size)
            datasets.append(OneHotEncodedDataset(dataset_path=Path(self.dataset_path) / dataset_name,
                                                 map_reader=map_reader,
                                                 map_size=self.map_size,
                                                 trajectory_min_len=self.trajectory_min_len,
                                                 trajectory_max_len=self.trajectory_max_len,
                                                 trajectory_sampling_rate=self.trajectory_sampling_rate,
                                                 config=self.config))
        dataloader = DataLoader(ConcatDataset(datasets), batch_size=self.batch_size, shuffle=False,
                                pin_memory=True, num_workers=self.num_workers, drop_last=True)
        print(f"Validation dataset length {len(dataloader)}")
        return dataloader


class TrajectoryDataModule(LightningDataModule):

    def __init__(self, map_size, map_type, map_path, dataset_path, trajectory_min_len, trajectory_max_len,
                 trajectory_sampling_rate,
                 n_negative_samples, negative_dist_thres,
                 batch_size=1, num_workers=0):
        super().__init__()
        self.map_size = map_size
        self.map_type = map_type
        self.map_path = Path(map_path)
        self.dataset_path = dataset_path
        self.trajectory_min_len = trajectory_min_len
        self.trajectory_max_len = trajectory_max_len
        self.trajectory_sampling_rate = trajectory_sampling_rate
        self.n_negative_samples = n_negative_samples
        self.negative_dist_thres = negative_dist_thres
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        datasets = []
        for dataset_name, map_name in train_maps(self.map_type).items():
            print(f"Adding {dataset_name} to train data set.")
            map_reader = MapReader(self.map_path / map_name, self.map_size)
            datasets.append(NegativeSamplingDataset(dataset_path=Path(self.dataset_path) / dataset_name,
                                                    map_reader=map_reader,
                                                    map_size=self.map_size,
                                                    trajectory_min_len=self.trajectory_min_len,
                                                    trajectory_max_len=self.trajectory_max_len,
                                                    trajectory_sampling_rate=self.trajectory_sampling_rate,
                                                    n_negative_samples=self.n_negative_samples,
                                                    negative_dist_thres=self.negative_dist_thres))
        dataloader = DataLoader(ConcatDataset(datasets), batch_size=self.batch_size, shuffle=True,
                                pin_memory=True, num_workers=self.num_workers)
        return dataloader

    def val_dataloader(self):
        datasets = []
        for dataset_name, map_name in val_maps(self.map_type).items():
            print(f"Adding {dataset_name} to val data set.")
            map_reader = MapReader(self.map_path / map_name, self.map_size)
            datasets.append(NegativeSamplingDataset(dataset_path=Path(self.dataset_path) / dataset_name,
                                                    map_reader=map_reader,
                                                    map_size=self.map_size,
                                                    trajectory_min_len=self.trajectory_min_len,
                                                    trajectory_max_len=self.trajectory_max_len,
                                                    trajectory_sampling_rate=self.trajectory_sampling_rate,
                                                    n_negative_samples=self.n_negative_samples,
                                                    negative_dist_thres=self.negative_dist_thres))
        dataloader = DataLoader(ConcatDataset(datasets), batch_size=self.batch_size, shuffle=False,
                                pin_memory=True, num_workers=self.num_workers)
        return dataloader


def get_metadata(track_path, data_range=None, sampling_rate=100):
    metadata = pd.read_csv(track_path / 'csv/extracted_data.csv')

    # metadata may be unordered, we have to order it by time and re-create the index
    metadata.sort_values(by=["timestamp"], inplace=True)
    # reset index and store original indexes in the 'index' column for debugging
    metadata.reset_index(inplace=True)

    if data_range:
        metadata = metadata.iloc[data_range].copy()

    metadata = metadata.dropna(subset=['gnss_latitude', 'gnss_longitude']).copy()

    metadata['velocity'] = np.sqrt(metadata['gnss_velocity_x'] ** 2 +
                                   metadata['gnss_velocity_y'] ** 2 +
                                   metadata['gnss_velocity_z'] ** 2)

    metadata = metadata[metadata['velocity'] > VELOCITY_MIN_TOLERANCE]
    metadata = metadata[metadata['velocity'] < VELOCITY_MAX_TOLERANCE]

    metadata = metadata.iloc[::sampling_rate, :]
    # create new index after filtering data, drop previous index
    metadata.reset_index(inplace=True, drop=True)
    return metadata[['index', 'timestamp', 'gnss_latitude', 'gnss_longitude']]

