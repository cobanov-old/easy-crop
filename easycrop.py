import os
import time
import pathlib
import argparse
import warnings
import numpy as np
from PIL import Image
from multiprocessing import Pool
from scipy.cluster.vq import kmeans2


warnings.simplefilter("ignore")


def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputfolder", metavar="INPUT_FOLDER", help="Input image folder"
    )
    parser.add_argument(
        "outputfolder", metavar="OUTPUT_FOLDER", help="Output image folder"
    )
    parser.add_argument(
        "--thumb", dest="thumbnail_size", type=int, default=200, help="Crop height"
    )
    return parser.parse_args()


options = parse_argument()


def calculate_kmeans(thumb):
    thumb_data = np.array(thumb.getdata()).astype(float)
    data = kmeans2(data=thumb_data, k=2, minit="points")[1]
    return data


def calculate_offset(arr, win):
    best = win**2
    offset = 0
    for i in range(len(arr) - win + 1):
        count = np.sum(arr[i : i + win])
        cur = np.abs((win**2 / 2) - count)
        if cur < best:
            best = cur
            offset = i
    return offset


def flatten(data, maj_size, min_size):
    return [
        np.sum(data[imin * maj_size + imaj] for imin in range(min_size))
        for imaj in range(maj_size)
    ]


def process(path):

    if path.endswith(".jpg") or path.endswith(".jpeg") or path.endswith(".png"):
        img = Image.open(os.path.join(options.inputfolder, path))
        width, height = img.size
        is_horizontal = width > height
        image_size = min(width, height)

        thumb = img.copy()
        thumb.thumbnail((options.thumbnail_size, options.thumbnail_size))
        twidth, theight = thumb.size
        min_size, major_size = min(twidth, theight), max(twidth, theight)

        data = calculate_kmeans(thumb)
        flat_data = flatten(data, major_size, min_size)

        thumb_offset = calculate_offset(flat_data, min_size)

        # rescaling
        offset = int(thumb_offset * (width / float(twidth)))

        if offset + image_size > width:
            offset = width - image_size

        if is_horizontal:
            crop = img.crop((offset, 0, offset + image_size, image_size))

        else:
            crop = img.crop((0, offset, image_size, offset + image_size))

        crop.save(f"{options.outputfolder}/{pathlib.Path(path).stem}_out.jpg")

    else:
        return None


def main():

    if not os.path.exists(options.outputfolder):
        os.makedirs(options.outputfolder, exist_ok=True)

    pool = Pool()  # Create a multiprocessing Pool
    paths = os.listdir(options.inputfolder)
    pool.map(process, paths)


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Time taken: {end - start}")
