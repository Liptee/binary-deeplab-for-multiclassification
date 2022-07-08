import os
import numpy as np
from glob import glob

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path, format_img, format_mask, only_mask = False):
    if not only_mask:
        X = sorted(glob(os.path.join(path, "imgs", f"*.{format_img}")))
        Y = sorted(glob(os.path.join(path, "masks", f"*.{format_mask}")))
        return X, Y
    if only_mask:
        Y = sorted(glob(os.path.join(path, "masks", f"*.{format_mask}")))
        return Y

def find_edges(img):
    mn = img.min()
    mx = img.max()
    return (mn, mx)

def line_contrasting(img):
    copy = np.array(img, copy=True)
    min, max = find_edges(img)
    copy = (img-min)/(max-min)*255
    return copy.astype('uint8')

def load_data_for_train(path):
    X = sorted(glob(os.path.join(path, "imgs", f"*.png")))
    Y = sorted(glob(os.path.join(path, "masks", f"*.png")))
    return X, Y
