from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Add, Lambda
import keras.backend as K
from keras.models import Model

def UNet_PRN(input_shape=(32, 32, 3)):
    inputs = Input(shape=input_shape)

    c1 = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
    c1 = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(p1)
    c2 = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    b = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu')(p2)
    b = Conv2D(128, kernel_size=3, strides=1, padding='same', activation='relu')(b)

    u1 = UpSampling2D((2, 2))(b)
    concat1 = Concatenate()([u1, c2])
    c3 = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(concat1)
    c3 = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(c3)

    u2 = UpSampling2D((2, 2))(c3)
    concat2 = Concatenate()([u2, c1])
    c4 = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(concat2)
    c4 = Conv2D(32, kernel_size=3, strides=1, padding='same', activation='relu')(c4)

    noise_mask = Conv2D(3, kernel_size=3, strides=1, padding='same')(c4)
    rectified = Add()([inputs, noise_mask])
    outputs = Lambda(lambda x: K.clip(x, 0.0, 1.0))(rectified)

    return Model(inputs, outputs, name="UNet_PRN")