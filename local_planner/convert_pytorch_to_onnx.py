import argparse

import yaml

from model.util import convert_to_onnx


def parse_arguments():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '--input-model-path',
        '-i',
        required=True
    )

    argparser.add_argument(
        '--model-config',
        '-t',
        required=True
    )

    argparser.add_argument(
        '--output-model-path',
        '-o',
        required=True
    )

    return argparser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    with open(args.model_config, "r") as f:
        config = yaml.safe_load(f)

    convert_to_onnx(args.input_model_path, args.output_model_path, config)

    print(f"The ONNX model has been created successfully.")
