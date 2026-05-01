from keras.layers import Input, Conv2D, Activation, Add, Lambda, GlobalAveragePooling2D, Multiply, Dense, Reshape
import keras.backend as K
from keras.models import Model

def se_block(input_tensor):
    num_channels = int(input_tensor.shape[-1])

    x = GlobalAveragePooling2D()(input_tensor)
    x = Dense(num_channels // 16, activation='relu')(x)
    x = Dense(num_channels, activation='sigmoid')(x)
    x = Reshape((1, 1, num_channels))(x)
    return Multiply()([input_tensor, x])


def PRN_SE(input_shape=(32, 32, 3)):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(inputs)
    
    for i in range(5):
        residual = x
        x = Conv2D(64, kernel_size=3, strides=1, padding='same', activation='relu')(x)
        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
        x = se_block(x)

        x = Add()([x, residual])
        x = Activation('relu')(x)
        
    x = Conv2D(16, kernel_size=3, strides=1, padding='same', activation='relu')(x)
    noise_mask = Conv2D(3, kernel_size=3, strides=1, padding='same')(x)
    rectified = Add()([inputs, noise_mask])
    outputs = Lambda(lambda x: K.clip(x, 0.0, 1.0))(rectified)
    
    return Model(inputs, outputs, name="PRN")