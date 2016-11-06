import sys
import os
from collections import defaultdict
import numpy as np
import scipy.misc


# def reverse_preprocess_input(x0):
#     x = x0 / 2.0
#     x += 0.5
#     x *= 255.
#     return x


def dataset(base_dir, w, h, preprocess_fn=None):
    d = defaultdict(list)
    for root, subdirs, files in os.walk(base_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            assert file_path.startswith(base_dir)
            suffix = file_path[len(base_dir):]
            suffix = suffix.lstrip("/")
            label = suffix.split("/")[0]
            d[label].append(file_path)

    tags = sorted(d.keys())

    processed_image_count = 0
    useful_image_count = 0

    X = []
    y = []

    for class_index, class_name in enumerate(tags):
        filenames = d[class_name]
        for filename in filenames:
            processed_image_count += 1
            img = scipy.misc.imread(filename)
            height, width, chan = img.shape
            assert chan == 3

#            aspect_ratio = float(max((height, width))) / min((height, width))
#            if aspect_ratio > 2:
#                continue
            # We pick the largest center square.
#            centery = height // 2
#            centerx = width // 2
#            radius = min((centerx, centery))
#            img = img[centery-radius:centery+radius, centerx-radius:centerx+radius]
            img = scipy.misc.imresize(img, size=(w, h), interp='bilinear')
            X.append(img)
            y.append(class_index)
            useful_image_count += 1
    print "processed %d, used %d" % (processed_image_count, useful_image_count)

    X = np.array(X).astype(np.float32)
#    X = X.transpose((0, 3, 1, 2))
    if preprocess_fn:
        X = preprocess_fn(X)
    y = np.array(y)

    perm = np.random.permutation(len(y))
    X = X[perm]
    y = y[perm]

    print "classes:"
    for class_index, class_name in enumerate(tags):
        print class_name, sum(y==class_index)
    print

    return X, y, tags