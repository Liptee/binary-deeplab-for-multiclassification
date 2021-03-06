import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from conf import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def SqueezeAndExcite(inputs, ratio=8):
    init = inputs
    filters = init.shape[-1]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    x = init * se
    return x

def ASPP(inputs, activation, dr1, dr2, dr3):
    shape = inputs.shape
    y1 = AveragePooling2D(pool_size=(shape[1],shape[2]))(inputs)
    y1 = Conv2D(256, 1, padding='same', use_bias=False)(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation("relu")(y1)
    y1 = UpSampling2D((shape[1],shape[2]),interpolation="bilinear")(y1)
   
    y2 = Conv2D(256, 1, padding='same', use_bias=False)(inputs)
    y2 = BatchNormalization()(y2)
    y2 = Activation("relu")(y2)

    y3 = Conv2D(256, 3, padding='same', use_bias=False, dilation_rate=int(dr1))(inputs)
    y3 = BatchNormalization()(y3)
    y3 = Activation("relu")(y3)

    y4 = Conv2D(256, 3, padding='same', use_bias=False, dilation_rate=int(dr2))(inputs)
    y4 = BatchNormalization()(y4)
    y4 = Activation("relu")(y4)

    y5 = Conv2D(256, 3, padding='same', use_bias=False, dilation_rate=int(dr3))(inputs)
    y5 = BatchNormalization()(y5)
    y5 = Activation("relu")(y5)

    y = Concatenate()([y1, y2, y3, y4, y5])
    y = Conv2D(256, 3, padding='same', use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation(activation)(y)

    return y

def deeplabv3_plus(shape, activation, act_aspp, dr1, dr2, dr3):
    inputs = Input(shape)

    encoder = ResNet50(
        weights="imagenet", 
        include_top=False, 
        input_tensor=inputs
        )

    image_features = encoder.get_layer("conv4_block6_out").output
    x_a = ASPP(image_features, act_aspp, dr1, dr2, dr3)
    x_a = UpSampling2D((4, 4), interpolation="bilinear")(x_a) ## x_a =ind= 128, 128, 256
    
    x_b = encoder.get_layer("conv2_block2_out").output
    x_b = Conv2D(filters=48, kernel_size=1, padding='same', use_bias=False)(x_b)
    x_b = BatchNormalization()(x_b)
    x_b = Activation("relu")(x_b)

    x = Concatenate()([x_a, x_b])
    x = SqueezeAndExcite(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x) 

    x = Conv2D(filters=256, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SqueezeAndExcite(x)   

    x = UpSampling2D((4, 4), interpolation="bilinear")(x)
    x = Conv2D(1, 1)(x)
    x = Activation(activation)(x)

    model = Model(inputs, x)
    return model

def check_model(act1, act2, dr1, dr2, dr3):
    model = deeplabv3_plus((H, W, 3), act1, act2, dr1, dr2, dr3)
    model.summary()