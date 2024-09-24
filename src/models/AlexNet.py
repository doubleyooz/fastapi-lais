from keras._tf_keras.keras.layers import Activation, Add, AveragePooling2D, Conv2D, MaxPooling2D, Flatten, Dense, Input, BatchNormalization, Dropout, ZeroPadding2D, Convolution2D, MaxPool2D

def AlexNet(shape = (32, 32, 3), classes = 4):
    x_input = Input(shape)
    x = ZeroPadding2D((3, 3))(x_input)

    # strides???
    #AlexNet
    x = Conv2D(96, kernel_size=(11,11), kernel_initializer='he_normal', strides=4,  activation='relu', input_shape=shape)(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), data_format=None)(x)
    x = Conv2D(256, kernel_size=(5,5), strides= 1, padding= 'same', activation= 'relu', kernel_initializer= 'he_normal')(x)
    x = MaxPooling2D(pool_size=(3,3), strides= (2,2), padding= 'valid', data_format= None)(x)
    x = Conv2D(384, kernel_size=(3,3), strides= 1, padding= 'same', activation= 'relu', kernel_initializer= 'he_normal')(x)
    x = Conv2D(384, kernel_size=(3,3), strides= 1, padding= 'same', activation= 'relu', kernel_initializer= 'he_normal') (x)
    x = Conv2D(256, kernel_size=(3,3), strides= 1, padding= 'same', activation= 'relu', kernel_initializer= 'he_normal') (x)
    x = MaxPooling2D(pool_size=(3,3), strides= (2,2), padding= 'valid', data_format= None)(x)
    x = Flatten()(x)
    x = Dense(4096, activation= 'relu')(x)
    x = Dense(4096, activation= 'relu')(x)
    x = Dense(1000, activation= 'relu')(x)
    x = Dense(classes, activation= 'softmax')(x)
    return x