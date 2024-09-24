
from keras._tf_keras.keras.layers import Activation, Add, AveragePooling2D, Conv2D, MaxPooling2D, Flatten, Dense, Input, BatchNormalization, Dropout, ZeroPadding2D, Convolution2D, MaxPool2D


def identity_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = Conv2D(filter, (3,3), padding = 'same')(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    # Layer 2
    x = Conv2D(filter, (3,3), padding = 'same')(x)
    x = BatchNormalization(axis=3)(x)
    # Add Residue
    x = Add()([x, x_skip])
    x = Activation('relu')(x)
    return x
