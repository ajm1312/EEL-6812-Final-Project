from keras.layers import Input, Conv2D, Activation, Add, Lambda
import keras.backend as K
from keras.models import Model

def PRN_Res(input_shape=(32, 32, 3)):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
    
    for i in range(5):
        residual = x
        x = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
        x = Add()([x, residual])
        x = Activation('relu')(x)
        
    x = Conv2D(16, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    noise_mask = Conv2D(3, kernel_size=3, strides=1, padding='same')(x)
    rectified = Add()([inputs, noise_mask])
    outputs = Lambda(lambda x: K.clip(x, 0.0, 1.0))(rectified)
    
    return Model(inputs, outputs, name="PRN")