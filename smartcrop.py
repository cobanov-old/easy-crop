#!/usr/bin/env python2

from sys import argv, exit
from PIL import Image
from numpy import array
from scipy.cluster.vq import kmeans2
import os
import time
from multiprocessing import Pool

tsiz = 200


def tobits(img):
    return kmeans2(array(img.getdata()).astype(float), 2, minit="points")[1]


def flatten(bits, majo, mino):
    return [
        sum(bits[imin * majo + imaj] for imin in range(mino)) for imaj in range(majo)
    ]


def get_offset(arr, win):
    best = win * win  # > max dist
    offset = 0
    for i in range(len(arr) - win + 1):
        count = sum(arr[i : i + win])
        # distance from half of the max count
        cur = abs((win * win / 2) - count)
        if cur < best:
            best = cur
            offset = i
    return offset


def main(src):
    if src.endswith(".jpg"):
        img = Image.open("./images/" + src)
        iw, ih = img.size
        xmajo, isiz = iw > ih, min(iw, ih)

        thumb = img.copy()
        thumb.thumbnail((tsiz, tsiz))
        tw, th = thumb.size
        mino, majo = min(tw, th), max(tw, th)

        bits = tobits(thumb)
        toffset = get_offset(flatten(bits, majo, mino), mino)

        # rescale offset
        ioffset = int(toffset * (iw / float(tw)))
        # correct the offset if it's out of bounds
        if ioffset + isiz > iw:
            ioffset = iw - isiz

        # print ('o: %s; (%s, %s)' % (ioffset, iw, ih))

        if xmajo:
            crop = img.crop((ioffset, 0, ioffset + isiz, isiz))
        else:
            crop = img.crop((0, ioffset, isiz, ioffset + isiz))

        crop.save(f"outputs/{src}_out.jpg")


if __name__ == "__main__":
    start = time.time()
    pool = Pool()  # Create a multiprocessing Pool
    pool.map(main, os.listdir("./images"))
    end = time.time()
    print(f"Time: {end - start}")
