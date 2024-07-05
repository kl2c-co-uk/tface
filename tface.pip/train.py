# Paths to training and validation directories
from dataset import dataset_main
dataset_norms = dataset_main()

train_image_dir = dataset_norms + '/train/images'
train_mask_dir = dataset_norms + '/train/masks'
validation_image_dir = dataset_norms + '/validation/images'
validation_mask_dir = dataset_norms + '/validation/masks'



# Create generators using ImageDataGenerator without resizing
print('2024-07-04; these settings arent working great ... or at all')


# this is the size of the input image
input_size=(1920, 1080)

# This is the size of the output heatmap
output_size = (192, 108)  


# ... now i shrink input!
input_size = (int(input_size[0] / 2), int(input_size[1] / 2))
output_size = (int(output_size[0] / 2), int(output_size[1] / 2))

batch_size = 16
epochs=10


# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import Resizing

# Define the model
import network
model = network.face_detector(input_size, output_size)

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Display the model summary
# model.summary()


# # Compile the model
# model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])


train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
	train_image_dir,
	target_size=input_size,
	batch_size=batch_size,
	class_mode=None,
	color_mode='rgb',
	seed=1)

train_mask_generator = train_datagen.flow_from_directory(
	train_mask_dir,
	target_size=output_size,
	batch_size=batch_size,
	class_mode=None,
	color_mode='grayscale',
	seed=1)

train_combined_generator = zip(train_generator, train_mask_generator)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
	validation_image_dir,
	target_size=input_size,
	batch_size=batch_size,
	class_mode=None,
	seed=1)

validation_mask_generator = validation_datagen.flow_from_directory(
	validation_mask_dir,
	target_size=output_size,
	batch_size=batch_size,
	class_mode=None,
	color_mode='grayscale',
	seed=1)

validation_combined_generator = zip(validation_generator, validation_mask_generator)

# Train the model
model.fit(
	train_combined_generator, steps_per_epoch=len(train_generator),
	validation_data=validation_combined_generator, validation_steps=len(validation_generator), 
	epochs=epochs)

# Save the model
model.save('target/face_detector.keras')

print('Model trained and saved!')
