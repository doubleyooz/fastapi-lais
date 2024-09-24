import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from models import ResNet34

class ClassifierController:
    
    train_path = 'drive/MyDrive/archive/Data/train'
    test_path = 'drive/MyDrive/archive/Data/test'
    height, width = 227, 227
    
    def create(self):
        modelRes = ResNet34(shape=(self.height, self.width, 3))


        batch_size = 32
        epochs=150

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        # Compile the model
        modelRes.compile(loss='categorical_crossentropy',  # And this
                    optimizer='adam',
                    metrics=['accuracy', 'f_score'])
    
    def train(self):
        # Create a data generator
        datagen  = ImageDataGenerator(
                        rescale=1./255,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        shear_range=0.1,
                        zoom_range=0.1)
        height, width = 227, 227
        # Load images from the 'train' directory
        train_generator = datagen.flow_from_directory(
            self.train_path,
            target_size=(height, width),
            batch_size=32,
            class_mode='categorical')
        
        test_generator = datagen.flow_from_directory(
            self.test_path,     # replace with your test folder
            target_size=(height, width),
            batch_size=32,
            class_mode='categorical')
        
        reduce_lr=ReduceLROnPlateau(
            monitor="accuracy",
            factor=0.4,
            patience=6,
            verbose=1
        )

        cp_filepath = 'drive/MyDrive/archive/training/cp.ckpt'

        cp_callback = ModelCheckpoint(filepath=cp_filepath,
                                                        save_weights_only=True,
                                                        verbose=1)

        callbacks=[reduce_lr, cp_callback]

        # Train the model
        history = modelRes.fit(
            train_generator,
            epochs=20,
            validation_data=train_generator,
            callbacks=callbacks
        )

    def load_model(self):
        
        modelRes = load_model('drive/MyDrive/archive/my_model.keras')
        # Show the model architecture
        modelRes.summary()
        
    def save_model(self):
        modelRes.save('drive/MyDrive/archive/my_model.keras')