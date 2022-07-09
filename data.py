import os
import cv2
import shutil
import numpy as np

from tqdm import tqdm
from conf import *

from sklearn.model_selection import train_test_split

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
    np.random.seed(42)
    data_path = "data"
    X, Y = load_data(data_path, format_img, format_mask)
    create_dir(f"data/labeled/{label}/masks")
    create_dir(f"data/labeled/{label}/imgs")

    reshape_and_save_imgs_and_masks(X, Y, label)

def create_test_data(f_imgs, f_masks):
    np.random.seed(42)
    tmp, _ = load_data("data/test", f_imgs, f_masks)
    X, Y = load_data("data", f_imgs, f_masks)
    sure = input(f"В тестовой папке уже содержится {len(tmp)} файлов, а в тренировочной выборке {len(X)} файлов. Вы уверены, что хотите продолжить?(y/n?): ")
    if sure == 'y':
        create_dir("data/test/imgs")
        create_dir("data/test/masks")
        coef = float(input('Введите коэфициент (от 0.01 до 0.99): '))
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = coef, random_state=42)
        for x, y in zip(x_test, y_test):
            i_name = x.split("imgs")[-1][1:]
            m_name = y.split("masks")[-1][1:]
            img_path = "data/test/imgs"
            mask_path = "data/test/masks"
            shutil.copyfile(x, f"{img_path}/{i_name}")
            shutil.copyfile(y, f"{mask_path}/{m_name}")
            os.remove(x)
            os.remove(y)

        X, Y = load_data("data", f_imgs, f_masks)
        tmp, _ = load_data("data/test", f_imgs, f_masks)
        print(f"Тренировочная выборка: {len(X)}; тестовая выборка: {len(tmp)}")

    else: return "cancel"

def recover_train_data(f_img, f_mask):
    X, Y = load_data("data/test", f_img, f_mask)

    for x, y in zip(X, Y):
        i_name = x.split("imgs")[-1][1:]
        m_name = y.split('masks')[-1][1:]
        print(i_name)
        print(m_name)
        print()
        img_path = "data/imgs"
        mask_path = "data/masks"
        shutil.copyfile(x, f"{img_path}/{i_name}")
        shutil.copyfile(y, f"{mask_path}/{m_name}")
        print(x)
        os.remove(x)
        os.remove(y)