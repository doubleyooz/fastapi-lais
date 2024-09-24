from keras._tf_keras.keras.layers import Activation, Add, AveragePooling2D, Conv2D, MaxPooling2D, Flatten, Dense, Input, BatchNormalization, Dropout, ZeroPadding2D, Convolution2D, MaxPool2D
import tensorflow as tf
from src.utils import convolutional_block, identity_block


def ResNet34(shape = (32, 32, 3), classes = 4):
    # Step 1 (Setup Input Layer)
    x_input = Input(shape)
    x = ZeroPadding2D((3, 3))(x_input)

    # Step 2 (Initial Conv layer along with maxPool)
    x = Conv2D(64, kernel_size=7, strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    # Define size of sub-blocks and initial filter size
    block_layers = [3, 4, 6, 3]
    filter_size = 64

    # Step 3 Add the Resnet Blocks
    for i in range(4):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size)
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size*2
            x = convolutional_block(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size)

    # Step 4 End Dense Network
    x = AveragePooling2D((2,2), padding = 'same')(x)
    x = Flatten()(x)
    # x = Dense(8192, activation='relu')(x)  # Added layer
    x = Dense(512, activation = 'relu')(x)
    x = Dense(classes, activation = 'softmax')(x)
    model = tf.keras.models.Model(inputs = x_input, outputs = x, name = "ResNet34")
    return model