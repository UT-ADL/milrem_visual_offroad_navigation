import cv2
import h5py
import os
import pickle

import numpy as np
import yaml
from PIL import Image
import io
import argparse
import tqdm

from gnm_train.visualizing.action_utils import gen_camera_matrix

"""
Adapted from: https://github.com/PrieureDeSion/drive-any-robot
"""


class ReconProcessor:

    def __init__(self) -> None:
        super().__init__()

        with open("gnm_train/data/data_config.yaml", "r") as f:
            data_config = yaml.safe_load(f)

        camera_metrics = data_config['recon']["camera_metrics"]
        fx = camera_metrics["camera_matrix"]["fx"]
        fy = camera_metrics["camera_matrix"]["fy"]
        cx = camera_metrics["camera_matrix"]["cx"]
        cy = camera_metrics["camera_matrix"]["cy"]
        self.camera_matrix = gen_camera_matrix(fx, fy, cx, cy)

        k1 = camera_metrics["dist_coeffs"]["k1"]
        k2 = camera_metrics["dist_coeffs"]["k2"]
        p1 = camera_metrics["dist_coeffs"]["p1"]
        p2 = camera_metrics["dist_coeffs"]["p2"]
        k3 = camera_metrics["dist_coeffs"]["k3"]
        self.dist_coeffs = np.array([k1, k2, p1, p2, k3, 0.0, 0.0, 0.0])

    def undistort(self, img):
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
        mapx, mapy = cv2.fisheye.initUndistortRectifyMap(self.camera_matrix, self.dist_coeffs[:4], np.eye(3), newcameramtx,
                                                         (w, h), 5)
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]
        return dst

    def process(self, input_dir, output_dir, num_trajs):
        recon_dir = os.path.join(input_dir, "recon_release")
        output_dir = output_dir

        # create output dir if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # get all the folders in the recon dataset
        filenames = os.listdir(recon_dir)
        if num_trajs >= 0:
            filenames = filenames[: num_trajs]

        # processing loop
        for filename in tqdm.tqdm(filenames, desc="Trajectories processed"):
            # extract the name without the extension
            traj_name = filename.split(".")[0]
            # load the hdf5 file
            try:
                h5_f = h5py.File(os.path.join(recon_dir, filename), "r")
            except OSError:
                print(f"Error loading {filename}. Skipping...")
                continue
            # extract the position and yaw data
            position_data = h5_f["jackal"]["position"][:, :2]
            yaw_data = h5_f["jackal"]["yaw"][()]
            # save the data to a dictionary
            traj_data = {"position": position_data, "yaw": yaw_data}
            traj_folder = os.path.join(output_dir, traj_name)
            os.makedirs(traj_folder, exist_ok=True)
            with open(os.path.join(traj_folder, "traj_data.pkl"), "wb") as f:
                pickle.dump(traj_data, f)
            # make a folder for the file
            if not os.path.exists(traj_folder):
                os.makedirs(traj_folder)
            # save the image data to disk
            for i in range(h5_f["images"]["rgb_left"].shape[0]):
                img = Image.open(io.BytesIO(h5_f["images"]["rgb_left"][i]))
                undistorted = self.undistort(np.array(img))
                img = Image.fromarray(undistorted)
                img.save(os.path.join(traj_folder, f"{i}.jpg"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # get arguments for the recon input dir and the output dir
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        help="path of the recon_dataset",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="datasets/recon/",
        type=str,
        help="path for processed recon dataset (default: datasets/recon/)",
    )
    # number of trajs to process
    parser.add_argument(
        "--num-trajs",
        "-n",
        default=-1,
        type=int,
        help="number of trajectories to process (default: -1, all)",
    )

    args = parser.parse_args()

    print(f"Starting to process Recon dataset from {args.input_dir}")
    recon_processor = ReconProcessor()
    recon_processor.process(args.input_dir, args.output_dir, args.num_trajs)
    print(f"Finished processing Recon dataset to {args.output_dir}")
