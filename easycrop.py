import os
import time
import warnings
import numpy as np
from PIL import Image
from multiprocessing import Pool
from scipy.cluster.vq import kmeans2

warnings.simplefilter("ignore")

thumbnail_size = 200
image_dir = "images2"


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
        img = Image.open(os.path.join(image_dir, path))
        width, height = img.size
        is_horizontal = width > height
        image_size = min(width, height)

        thumb = img.copy()
        thumb.thumbnail((thumbnail_size, thumbnail_size))
        twidth, theight = thumb.size
        min_size, major_size = min(twidth, theight), max(twidth, theight)

        data = calculate_kmeans(thumb)
        flat_data = flatten(data, major_size, min_size)

        thumb_offset = calculate_offset(flat_data, min_size)

        # rescaling
        offset = int(thumb_offset * (width / float(twidth)))
        # correct the offset if it's out of bounds
        if offset + image_size > width:
            offset = width - image_size

        if is_horizontal:
            crop = img.crop((offset, 0, offset + image_size, image_size))

        else:
            crop = img.crop((0, offset, image_size, offset + image_size))

        crop.save(f"./outputs/{path}_out.jpg")

    else:
        return None


def main():
    start = time.time()
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    pool = Pool()  # Create a multiprocessing Pool
    pool.map(process, os.listdir(image_dir))
    end = time.time()
    print(f"Time: {end - start}")


if __name__ == "__main__":
    main()
