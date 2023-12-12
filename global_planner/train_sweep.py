import argparse

from train_segment import train, read_config


def main():
    config = read_config("config/default_segment.yaml", "distance-map-sweep")

    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--mask_type",
        type=str,
        required=True,
    )
    argparser.add_argument(
        "--trajectory_max_length",
        type=int,
        required=True,
    )
    args = argparser.parse_args()

    print("mask_type: ", args.mask_type)
    config["mask_type"] = args.mask_type

    print("trajectory_max_length: ", args.trajectory_max_length)
    config["trajectory_max_length"] = args.trajectory_max_length

    train(config)


if __name__ == "__main__":
    main()
