import csv

import cv2
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Conv2D, Convolution2D, Cropping2D
from keras.layers.core import Activation, Dense, Dropout, Flatten, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Optimizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def read_data_from_csv():
    samples = []
    with open('./data/driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        samples = [line for line in reader]
    return samples


def generate_data(input_samples, batch_size):
    bias = 0.25
    input_samples = shuffle(input_samples)
    num_samples = len(input_samples)
    while True:
        for offset in range(0, num_samples, batch_size):
            batch_samples = input_samples[offset:offset+batch_size]
            images = []
            steering_angles = []
            for batch_sample in batch_samples:
                center_img = cv2.imread(batch_sample[0])
                left_img = cv2.imread(batch_sample[1])
                right_img = cv2.imread(batch_sample[2])
                center_image_flipped = cv2.flip(center_img, 1)
                steering_angle = float(batch_sample[3])

                images.append(center_img)
                steering_angles.append(steering_angle)

                images.append(left_img)
                steering_angles.append(steering_angle + bias)

                images.append(right_img)
                steering_angles.append(steering_angle - bias)

                images.append(center_image_flipped)
                steering_angles.append(-1.0 * steering_angle)
            X = np.array(images)
            y = np.array(steering_angles)
            yield shuffle(X, y)


batch_size = 128
dropout_rate = 0.30

samples = read_data_from_csv()
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
generate_train_data = generate_data(train_samples, batch_size)
generate_validation_data = generate_data(validation_samples, batch_size)


# Implement the model based of of NVIDIA 'End to End Learning for Self-Driving Cars'
model = Sequential()
# Normalize input
model.add(Lambda(lambda img: (img / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(dropout_rate))
model.add(Dense(50))
model.add(Dropout(dropout_rate))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

print(model.summary())

# Stop training the model val_loss has not imporved after 3 epochs
early_stop_callback = EarlyStopping(
    monitor='val_loss', patience=3)

# Save the best performing version of the model to model.h5
model_checkpoint_callback = ModelCheckpoint(
    filepath='model.h5',
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose=1)

# Train
model.fit_generator(
    generate_train_data,
    steps_per_epoch=np.ceil(len(train_samples)/batch_size),
    validation_data=generate_validation_data,
    validation_steps=np.ceil(
        len(validation_samples)/batch_size),
    callbacks=[early_stop_callback, model_checkpoint_callback],
    epochs=100, verbose=1)
