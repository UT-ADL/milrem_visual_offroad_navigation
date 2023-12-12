import numpy as np
import pandas as pd
import rospy
import torch
from PIL import Image
from pytransform3d import rotations as pr
from pytransform3d import transformations as pt
from tf.transformations import euler_from_quaternion
from torch.utils.data import Dataset
from torchvision import transforms

from gnm_train.data.data_utils import img_path_to_data


def int_to_ros_time(int_time):
    ros_time = str(int_time)
    return rospy.Time(int(ros_time[0:10]), int(ros_time[10:]))


IMAGE_ASPECT_RATIO = (4 / 3)  # all images are centered cropped to a 4:3 aspect ratio in training
VISUALIZATION_IMAGE_SIZE = (120, 160)


class MilremDataset(Dataset):
    # TODO: move to yaml config
    FRAMES_PER_SECOND = 15.0
    # Sampled waypoint temporal range in frames
    MIN_WAYPOINT_DISTANCE = 5
    MAX_WAYPOINT_DISTANCE = 75

    # Used to filter out bad data
    POSITIONAL_TOLERANCE = 1.0
    VELOCITY_MIN_TOLERANCE = 0.05
    VELOCITY_MAX_TOLERANCE = 2.5

    def __init__(self, dataset_dir, transform, img_size, aspect_ratio, frame_range=None, **kwargs):
        self.dataset_dir = dataset_dir
        self.transform = transform
        self.img_size = img_size
        self.aspect_ratio = aspect_ratio

        self.context_size = kwargs["context_size"]
        self.waypoint_spacing = 4
        #self.waypoint_spacing = kwargs["waypoint_spacing"]
        self.pred_trajectory_length = kwargs["len_traj_pred"]

        self.metadata = self.get_metadata(frame_range)

        # length + 1 as we need to include current position in temporal length calculation
        self.context_temporal_length = (self.context_size + 1) * self.waypoint_spacing
        self.trajectory_temporal_length = (self.pred_trajectory_length + 1) * 15
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

        # # Subsample rows
        # # filtered_metadata['include'] = False
        # currtime = int_to_ros_time(filtered_metadata.iloc[0]['timestamp']).to_sec()
        # for index, row in filtered_metadata.iterrows():
        #     t = int_to_ros_time(row['timestamp'])
        #     # print(t.to_sec() - currtime)
        #     if (t.to_sec() - currtime) >= 1.0 / self.RATE:
        #         filtered_metadata.loc[index, 'include'] = True
        #         currtime = t.to_sec()
        #
        # # print("len1: ", len(filtered_metadata))
        # filtered_metadata = filtered_metadata[filtered_metadata['include'] == True]

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

        distance_label = torch.tensor([waypoint_distance]).float()
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
        #print("context_ids: ", context_ids)
        for i in context_ids:
            img_path = self.dataset_dir / "images" / self.metadata.iloc[i]['image_name']
            obs_image, transf_obs_image = img_path_to_data(str(img_path), self.transform, self.img_size, self.aspect_ratio)
            observation_images.append(transf_obs_image)

        observations = torch.cat(observation_images)
        return observations

    def sample_waypoint(self, current_pos_loc):
        #waypoint_distance = random.randint(self.MIN_WAYPOINT_DISTANCE, self.MAX_WAYPOINT_DISTANCE)
        #wp_idx = min(current_pos_loc + waypoint_distance, len(self.metadata) - 1)
        distance_frames = 75
        wp_idx = current_pos_loc + distance_frames
        waypoint_distance = (distance_frames / 4)  # TODO: fix this, should divide with spacing
        waypoint_data = self.metadata.iloc[wp_idx]
        img_path = self.dataset_dir / "images" / waypoint_data["image_name"]
        image, transf_image = img_path_to_data(str(img_path), self.transform, self.img_size, self.aspect_ratio)
        return waypoint_data, transf_image, waypoint_distance

    def calculate_current_pos_loc(self, idx):
        # All ids are shifted to provided long enough context in the beginning of the dataset
        return self.context_size * self.waypoint_spacing + idx

    def load_trajectory(self, current_pos_loc):
        vehicle_row = self.metadata.iloc[current_pos_loc]

        waypoint_ids = np.arange(15, self.trajectory_temporal_length, 15) + current_pos_loc
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

    def __init__(self, dataset_dir, frame_range=None, **kwargs):
        transform = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        image_transform = transforms.Compose(transform)
        aspect_ratio = kwargs["image_size"][0] / kwargs["image_size"][1]

        super().__init__(dataset_dir, image_transform, kwargs['image_size'], aspect_ratio, frame_range, **kwargs)

    def __getitem__(self, idx):
        observations, sampled_waypoint, labels, data = super().__getitem__(idx)

        current_pos_idx = data["idx"]
        img_path = self.dataset_dir / "images" / self.metadata.iloc[current_pos_idx]['image_name']
        current_obs_img = Image.open(str(img_path))

        wp_pos_idx = data["wp_idx"]
        img_path = self.dataset_dir / "images" / self.metadata.iloc[wp_pos_idx]['image_name']
        waypoint_img = Image.open(str(img_path))

        return observations, sampled_waypoint, labels, data, current_obs_img, waypoint_img
