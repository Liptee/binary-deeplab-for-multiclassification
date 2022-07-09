import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tools import (create_dir,
load_data_for_train)

import numpy as np
import cv2
from glob import glob
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall
from tensorflow.keras.metrics import Precision
from model import deeplabv3_plus
from metrics import dice_loss
from metrics import dice_coef
from metrics import iou
from conf import *

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = x/255.0
    x = x.astype(np.float32)
    return x

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(X, Y, batch=2):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset


def train(label, act1, act2, dr1, dr2, dr3, format_img, format_mask):

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
	
    np.random.seed(42)
    tf.random.set_seed(42)

    create_dir("files")

    lr = 1e-4
    model_path = os.path.join(f"files", f"model_{label}.h5")
    csv_path = os.path.join("files", f"data_{label}.csv")

    train_path = f"data/labeled/{label}"

    train_x, train_y = load_data_for_train(train_path)
    print(len(train_x))

    print(f"Train: {len(train_x)} - {len(train_y)}")

    train_dataset = tf_dataset(train_x, train_y, batch)

    model = deeplabv3_plus((H, W, 3), act1, act2, dr1, dr2, dr3)
    model.compile(loss=dice_loss, optimizer=Adam(lr), metrics=[dice_coef, iou, Recall(), Precision()])

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_model=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        TensorBoard(),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False),
    ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        callbacks=callbacks
    )