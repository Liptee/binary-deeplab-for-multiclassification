import os
import cv2
import numpy as np

from glob import glob
from tqdm import tqdm

from data import create_dirs_for_every_label_of_masks
from model import check_model
from train import train

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

format_img = "jpg"
format_mask = "png"

if input("Используем стандартные настройки?") == "N":
    format_img = input("Введите формат изображений: ")
    format_mask = input("Введите формат масок: ")

print("Идет сканирование масок")
masks = sorted(glob(os.path.join("data", "masks", f"*.{format_mask}")))
imgs = sorted(glob(os.path.join("data", "imgs", f"*.{format_img}")))
labels = []
for y in tqdm(masks):
    m = np.unique(cv2.imread(y, cv2.IMREAD_GRAYSCALE))
    for label in m:
        if label not in labels:
            labels.append(label)
num_labels = len(np.unique(labels))

print(f"Количество масок: {len(masks)}")
print(f"Количество изображений: {len(imgs)}")
print(f"Количество классов: {num_labels}")

command = ""
while command != "exit":
    print()
    command = input(":::")
    if command == "aug":
        print("Начинается подготовка данных")
        for label in range(num_labels):
            create_dirs_for_every_label_of_masks(label, format_img, format_mask)

    
    if command == "check":
        check_object = input("Что вы хотите проверить(model?): ")
        if check_object == "model":
            q = input("Stardart?(y or n):::")
            if q == "y":
                check_model("sigmoid", "sigmoid", 6, 12, 18)
            if q == "n":
                act1 = input("choose act1(Standart: sigmoid):::")
                act2 = input("choose act2(Standart: sigmoid):::")
                dr1 = input("choose dilation rate 1(Standart: 6):::")
                dr2 = input("choose dilation rate 2(Standart: 12):::")
                dr3 = input("choose dilation rate 3(Standart: 18):::")
                try:
                    check_model(act1, act2, dr1, dr2, dr3)
                except:
                    print("Недопустимое значение параметров")
                    pass

    if command == "train":
        i = input("choose label")
        q = input("Stardatr?(y or n):::")
        if q == "y":
            train(int(i), "sigmoid", "sigmoid", 6, 12, 18, format_img, format_mask)
        if q == "n":
            act1 = input("choose act1(Standart: sigmoid):::")
            act2 = input("choose act2(Standart: sigmoid):::")
            dr1 = input("choose dilation rate 1(Standart: 6):::")
            dr2 = input("choose dilation rate 2(Standart: 12):::")
            dr3 = input("choose dilation rate 3(Standart: 18):::")
            train(int(i), act1, act2, dr1, dr2, dr3, format_img, format_mask)