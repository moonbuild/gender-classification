import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras import models
from keras import layers
import keras.preprocessing as kp
from tensorflow.keras.models import save_model
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras import optimizers

train_location = '/home/tiltedcrown/Downloads/datasets/1_one/Training/'
validation_location = '/home/tiltedcrown/Downloads/datasets/1_one/Validation/'
# for WINDOWS train_location = r'C:\Users\Duck\path\to\train

train_datagen  =ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_location,
    target_size=(250,250),
    batch_size=48,
    class_mode='binary'
)

valid_gen = test_datagen.flow_from_directory(
    validation_location,
    target_size=(250,250),
    batch_size=48,
    class_mode='binary'
)

kernel_size=(3,3)

model=models.Sequential()

model.add(
    layers.Conv2D(
        32, kernel_size,
        activation='relu', input_shape=(250,250,3),
        kernel_regularizer=regularizers.l2(0.001), padding="VALID")
)
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64, kernel_size, activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, kernel_size, activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128, kernel_size, activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, kernel_size, activation='relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop', loss= 'binary_crossentropy', metrics=['acc'])

history=model.fit(
    train_gen, steps_per_epoch=70, epochs=30,
    validation_data=valid_gen, validation_steps=50
)

save_model(model, 'model-gender.h5')
model.save_weights('my_model_weights.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'ro', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('accuracy_plot.png')

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'ro', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.savefig('loss_plot.png')

plt.figure()
