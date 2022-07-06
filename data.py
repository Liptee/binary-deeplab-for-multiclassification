import os
import cv2
import numpy as np

from tqdm import tqdm
from conf import *

from tools import (create_dir,
load_data,
line_contrasting)

save_path = "data/labeled"

def reshape_and_save_imgs_and_masks(imgs, masks, label):
    for x, y in tqdm(zip(imgs, masks), total=len(imgs)):
        name = (x.split("imgs")[-1].split(".")[0])[1:]
        mask_name = (y.split("masks")[-1].split(".")[0])[1:] 
        if name != mask_name:
            print("Несовпадение названий изображений и масок")
            print('Проверьте целостность датасета')
            break

        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)

        X = [x]
        Y = [y]

        for img, mask in zip(X, Y):
            img = cv2.resize(img, (W, H))
            mask = cv2.resize(mask, (W, H))
            # img = line_contrasting(img) #линейное контрастирование. Возможно неадоптировано под цветные изображения
            for i in range(len(mask)):
                for j in range(len(mask[0])):
                    if mask[i][j] == label: mask[i][j] = 1
                    else: mask[i][j] = 0
        
        new_name = f"{name}_label{label}.png"
        image_path = os.path.join(save_path, f"{label}", "imgs", new_name)
        mask_path = os.path.join(save_path, f"{label}", "masks", new_name)

        cv2.imwrite(image_path, img)
        cv2.imwrite(mask_path, mask)


def create_dirs_for_every_label_of_masks(label, format_img, format_mask):
    # np.random.seed(42)
    data_path = "data"
    X, Y = load_data(data_path, format_img, format_mask)
    create_dir(f"data/labeled/{label}/masks")
    create_dir(f"data/labeled/{label}/imgs")

    reshape_and_save_imgs_and_masks(X, Y, label)