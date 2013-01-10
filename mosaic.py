#!/usr/bin/env python

import Image
from numpy import *
from scipy.spatial import KDTree
import glob
import json
import os
import logging
from collections import deque
import argparse

THUMB_SIZE = (100, 100)
THUMB_FILTER = Image.BILINEAR
THUMB_CACHE = "cache.json"
MOSAIC_SIZE = (200, 200)


def dominant_color(im):
    pixels = array(im.getdata())
    color = median(pixels, axis=0).astype(int)
    return tuple(color)


def process_images(path):
    if os.path.exists(THUMB_CACHE):
        cache = json.load(open(THUMB_CACHE))
    else:
        cache = {}

    thumbs_dir = os.path.join(path, "thumbs")
    if not os.path.exists(thumbs_dir):
        os.mkdir(thumbs_dir)

    names = glob.glob("%s/*.[JjPp][PpNn][Gg]" % path)
    ni = 1
    for n in names:
        ni += 1
        if n in cache:
            continue
        logging.info("Processing image %s, %d of %d" % (n, ni, len(names)))

        im = Image.open(n)
        im.thumbnail(THUMB_SIZE, THUMB_FILTER)
        color = dominant_color(im)

        tn = "%s/%s" % (thumbs_dir, os.path.basename(n))

        # Save the "squared thumbnail"
        nimg = Image.new("RGB", THUMB_SIZE, color)
        box = (array(THUMB_SIZE) - array(im.size)) / 2
        nimg.paste(im, tuple(box))
        nimg.save(tn)

        #
        cache[n] = {"color": color, "thumbnail": tn}
        json.dump(cache, open(THUMB_CACHE, "w"))

    return cache


class Palette(object):
    def __init__(self, images):
        self.cache = {}
        self.kdtree = KDTree([v["color"] for v in images.itervalues()])

        self.pmap = {}
        for (img_path, i) in images.iteritems():
            color = tuple(i["color"])
            if not color in self.pmap:
                self.pmap[color] = deque()
            self.pmap[color].append(i["thumbnail"])

    def get_closest_color(self, c):
        dist, idx = self.kdtree.query(c)
        return tuple(self.kdtree.data[idx])

    def get_image(self, color):
        variants = self.pmap[color]
        # Try to use different images
        iname = variants.popleft()
        variants.append(iname)

        img = self.cache.get(iname)
        if not img:
            img = Image.open(iname)
            self.cache[iname] = img
        return img

    def get_closest_image(self, c):
        nc = self.get_closest_color(c)
        return self.get_image(nc)


def dither(im, palette):
    pix = im.load()

    def adjust(x, y, v):
        if x < 0 or x >= im.size[0] or y >= im.size[1]:
            return
        fp = array(pix[x, y]).astype(float)
        pix[x, y] = tuple(around((fp + v)).astype(int))

    logging.info("Dithering the image")
    for y in xrange(0, im.size[1]):
        for x in xrange(0, im.size[0]):
            old = pix[x, y]
            new = palette.get_closest_color(old)
            pix[x, y] = new
            quant_error = array(old) - array(new)
            adjust(x + 1, y, quant_error * (7 / 16.0))
            adjust(x - 1, y + 1, quant_error * (3 / 16.0))
            adjust(x, y + 1, quant_error * (5 / 16.0))
            adjust(x + 1, y + 1, quant_error * (1 / 16.0))
    return im


def make_mosaic(im, palette):
    (sx, sy) = THUMB_SIZE
    nsize = (sx * im.size[0], sy * im.size[1])
    ni = Image.new("RGB", nsize)
    pix = im.load()
    total = im.size[0] * im.size[1]
    for y in xrange(0, im.size[1]):
        logging.info("Making mosaic %d of %d", y * im.size[0], total)
        for x in xrange(0, im.size[0]):
            p = palette.get_closest_image(pix[x, y])
            ni.paste(p, (sx * x, sy * y))
    return ni


def save_result(img, fname):
    logging.info(fname)
    img.save(fname)


def show_result(img):
    logging.info("showing")
    img.show()


def size(s):
    try:
        x, y = map(int, s.split('x'))
        return (x, y)
    except:
        raise argparse.ArgumentTypeError("Must be NxN")


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--images", default="images")
    parser.add_argument("-o", "--output", default="result.jpg")
    parser.add_argument("-s", "--size", default=MOSAIC_SIZE, type=size)
    parser.add_argument("input", metavar="input.jpg")
    args = parser.parse_args()

    cache = process_images(args.images)
    palette = Palette(cache)

    src = Image.open(args.input)
    src.thumbnail(args.size, THUMB_FILTER)
    dither(src, palette)
    res = make_mosaic(src, palette)
    save_result(res, args.output)
    #show_result(res)

if __name__ == "__main__":
    main()
