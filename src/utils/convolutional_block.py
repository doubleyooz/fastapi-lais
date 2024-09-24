
from keras._tf_keras.keras.layers import Activation, Add, AveragePooling2D, Conv2D, MaxPooling2D, Flatten, Dense, Input, BatchNormalization, Dropout, ZeroPadding2D, Convolution2D, MaxPool2D

def convolutional_block(x, filter):
    # copy tensor to variable called x_skip
    x_skip = x
    # Layer 1
    x = Conv2D(filter, (3,3), padding = 'same', strides = (2,2))(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation('relu')(x)
    # Layer 2
    x = Conv2D(filter, (3,3), padding = 'same')(x)
    x = BatchNormalization(axis=3)(x)
    # Processing Residue with conv(1,1)
    x_skip = Conv2D(filter, (1,1), strides = (2,2))(x_skip)
    # Add Residue
    x = Add()([x, x_skip])
    x = Activation('relu')(x)
    return x