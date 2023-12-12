import argparse
import os

from data.milrem_tracks import CLEANED_TRACKS
from data.process_milrem import create_track_name
from data.recon_data_split import remove_files_in_dir

DATA_SPLITS_DIR = "data/data_splits"


def create_train_split(dataset_name):

    train_dir = os.path.join(DATA_SPLITS_DIR, dataset_name, "train")
    test_dir = os.path.join(DATA_SPLITS_DIR, dataset_name, "test")
    for dir_path in [train_dir, test_dir]:
        if os.path.exists(dir_path):
            print(f"Clearing files from {dir_path} for new data split")
            remove_files_in_dir(dir_path)
        else:
            print(f"Creating {dir_path}")
            os.makedirs(dir_path)

    with open(os.path.join(train_dir, "traj_names.txt"), "w") as f:
        for t in CLEANED_TRACKS:
            if t[2] == 'train':
                f.write(create_track_name(t) + "\n")

    with open(os.path.join(test_dir, "traj_names.txt"), "w") as f:
        for t in CLEANED_TRACKS:
            if t[2] == 'val':
                f.write(create_track_name(t) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-name", "-d", help="Name of the dataset", default="milrem"
    )
    args = parser.parse_args()

    print("Creating Milrem data split...")
    create_train_split(args.dataset_name)
    print("Done")
