import argparse
import os
import pickle
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import rospy
import tqdm
from tf.transformations import euler_from_quaternion

from data.milrem_tracks import CLEANED_TRACKS

POSITIONAL_TOLERANCE = 1.0
VELOCITY_MIN_TOLERANCE = 0.05
VELOCITY_MAX_TOLERANCE = 2.5
RATE = 4.0


def main(input_dir, output_dir):
    for track in tqdm.tqdm(CLEANED_TRACKS, desc="Trajectories processed"):
        process_dataset(input_dir, output_dir, track)


def process_dataset(input_idr, output_dir, track):
    track_path = Path(input_idr) / track[0]
    data_range = track[1]

    metadata = get_metadata(track_path, data_range)
    position_data = metadata[['camera_position_x', 'camera_position_y']].to_numpy()
    yaw_data = metadata.apply(lambda row: calculate_yaw(row),  axis = 1).to_numpy()
    traj_data = {"position": position_data, "yaw": yaw_data}

    track_name = create_track_name(track)
    traj_folder = os.path.join(output_dir, track_name)
    os.makedirs(traj_folder, exist_ok=True)

    # create trajectory pickle
    with open(os.path.join(traj_folder, "traj_data.pkl"), "wb") as f:
        pickle.dump(traj_data, f)

    # copy images
    image_names = metadata['image_name'].to_numpy()
    for i, image_name in enumerate(image_names):
        shutil.copyfile(track_path / 'images' / image_name, Path(traj_folder) / f"{i}.jpg")


def get_metadata(track_path, data_range=None):
    metadata = pd.read_csv(track_path / 'csv/extracted_data.csv')

    # metadata may be unordered, we have to order it by time and re-create the index
    metadata.sort_values(by=["timestamp"], inplace=True)
    # reset index and store original indexes in the 'index' column for debugging
    metadata.reset_index(inplace=True)

    if data_range:
        metadata = metadata.iloc[data_range].copy()

    metadata['velocity'] = np.sqrt(metadata['gnss_velocity_x'] ** 2 +
                                   metadata['gnss_velocity_y'] ** 2 +
                                   metadata['gnss_velocity_z'] ** 2)

    filtered_data = metadata.dropna(subset=['camera_position_x', 'camera_position_y', 'camera_position_z']).copy()

    filtered_data['diff_x'] = filtered_data['camera_position_x'].diff().abs()
    filtered_data['diff_y'] = filtered_data['camera_position_y'].diff().abs()
    filtered_data['diff_z'] = filtered_data['camera_position_z'].diff().abs()

    filtered_metadata = filtered_data[(filtered_data['diff_x'] < POSITIONAL_TOLERANCE) &
                                 (filtered_data['diff_y'] < POSITIONAL_TOLERANCE) &
                                 (filtered_data['diff_z'] < POSITIONAL_TOLERANCE)]

    filtered_metadata = filtered_metadata[filtered_metadata['velocity'] > VELOCITY_MIN_TOLERANCE]
    filtered_metadata = filtered_metadata[filtered_metadata['velocity'] < VELOCITY_MAX_TOLERANCE]


    # Subsample rows
    #filtered_metadata['include'] = False
    currtime = int_to_ros_time(filtered_metadata.iloc[0]['timestamp']).to_sec()
    for index, row in filtered_metadata.iterrows():
        t = int_to_ros_time(row['timestamp'])
        #print(t.to_sec() - currtime)
        if (t.to_sec() - currtime) >= 1.0 / RATE:
            filtered_metadata.loc[index, 'include'] = True
            currtime = t.to_sec()

    #print("len1: ", len(filtered_metadata))
    filtered_metadata = filtered_metadata[filtered_metadata['include'] == True]
    #print("len2: ", len(filtered_metadata))
    # create new index after filtering data, drop previous index
    filtered_metadata.reset_index(inplace=True, drop=True)
    return filtered_metadata


def create_track_name(track):
    start = track[1].start
    if not start:
        start = 0

    stop = track[1].stop
    if not stop:
        stop = 0

    return f"{track[0]}_{start}_{stop}_{track[2]}"


def create_train_split(tracks):
    for t in tracks:
        if t[2] == 'train':
            print(create_track_name(t))


def create_validation_split(tracks):
    for t in tracks:
        if t[2] == 'val':
            print(create_track_name(t))


def int_to_ros_time(int_time):
    ros_time = str(int_time)
    return rospy.Time(int(ros_time[0:10]), int(ros_time[10:]))


def stamp_to_datetime(ros_time):
    ros_time = int_to_ros_time(ros_time)
    epoch = datetime(1970, 1, 1)  # ROS time epoch
    time_difference = timedelta(seconds=ros_time.secs, microseconds=ros_time.nsecs / 1000)
    python_datetime = epoch + time_difference
    return python_datetime


def calculate_yaw(row):
    quaternion = [
        row['camera_orientation_x'], row['camera_orientation_y'],
        row['camera_orientation_z'], row['camera_orientation_w']
    ]
    roll, pitch, yaw = euler_from_quaternion(quaternion)
    return yaw


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # get arguments for the recon input dir and the output dir
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        help="Path to the Milrem dataset",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="datasets/milrem/",
        type=str,
        help="path do processed dataset (default: datasets/milrem/)",
    )

    args = parser.parse_args()
    print(f"Starting to process Milrem dataset from {args.input_dir}")
    main(args.input_dir, args.output_dir)
    print(f"Finished processing Milrem dataset to {args.output_dir}")
