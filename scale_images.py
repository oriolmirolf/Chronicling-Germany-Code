import argparse
import os
from typing import Optional

from PIL import Image
from PIL.Image import BICUBIC
from tqdm import tqdm


def rescale(args: argparse.Namespace):
    data_path = adjust_path(args.data_path)
    output_path = adjust_path(args.output_path)
    extension = args.extension

    file_names = [
        f[:-4] for f in os.listdir(data_path) if f.endswith(extension)
    ]
    for name in tqdm(file_names, desc="Rescaling images", unit="image"):
        image = Image.open(data_path + name + extension).convert("RGB")
        shape = int(image.size[0] * args.scale), int(image.size[1] * args.scale)
        image = image.resize(shape, resample=BICUBIC)
        image.save(output_path + name + extension)


def adjust_path(path: Optional[str]) -> Optional[str]:
    """
    Make sure, there is a slash at the end of a (folder) spath string.
    """
    return path if not path or path[-1] == '/' else path + '/'


def get_args() -> argparse.Namespace:
    """defines arguments"""
    parser = argparse.ArgumentParser(description="Rescaling script")
    parser.add_argument(
        "--data-path",
        "-d",
        type=str,
        default="data/",
        help="Data path for images.",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        default="output/",
        help="Output path for images.",
    )
    parser.add_argument(
        "--extension",
        type=str,
        default=".jpg",
        help="Image extension.",
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=float,
        default=1,
        help="Scaling factor for images.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parameter_args = get_args()

    if not os.path.exists(f"{parameter_args.output_path}"):
        os.makedirs(f"{parameter_args.output_path}")

    rescale(parameter_args)
