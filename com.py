import os
import cv2
import numpy as np

from glob import glob
from tqdm import tqdm

from data import create_dirs_for_every_label_of_masks

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