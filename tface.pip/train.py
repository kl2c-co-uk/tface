# Paths to training and validation directories
from dataset import dataset_main
dataset_norms = dataset_main()

train_image_dir = dataset_norms + '/train/images'
train_mask_dir = dataset_norms + '/train/masks'
validation_image_dir = dataset_norms + '/validation/images'
validation_mask_dir = dataset_norms + '/validation/masks'



# Create generators using ImageDataGenerator without resizing
print('2024-07-04; these settings arent working great ... or at all')
input_size=(1920, 1080, 3)
resize_scale = 0.25 # UGGHUU i hate myself for this but yeah
batch_size = 16
target_size = (192, 108)  # This is the size of the output heatmap
epochs=10


# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import Resizing

# Define the model for 1920x1080 input and 192x108 output
def simple_cnn_model():
	inputs = Input(input_size)

	# resize the first one

	# peter; is thia bsd?
	resized = Conv2D(4, 3, activation='relu', padding='same')(inputs)
	

	
	# Example simple CNN architecture
	conv1 = Conv2D(64, 3, activation='relu', padding='same')(resized)
	conv2 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
	pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
	
	conv3 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
	conv4 = Conv2D(128, 3, activation='relu', padding='same')(conv3)
	pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)
	
	conv5 = Conv2DTranspose(64, 3, strides=(2, 2), activation='relu', padding='same')(pool2)
	conv6 = Conv2DTranspose(32, 3, strides=(2, 2), activation='relu', padding='same')(conv5)
	
	outputs = Conv2D(1, 1, activation='sigmoid', padding='same')(conv6)  # Adjust activation and padding as needed
	
	model = Model(inputs=inputs, outputs=outputs)
	return model

# Define the model
model = simple_cnn_model()

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])


train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
	train_image_dir,
	target_size=target_size,  # Input size remains 1920x1080
	batch_size=batch_size,
	class_mode=None,  # Because we will use custom loss
	seed=1)

train_mask_generator = train_datagen.flow_from_directory(
	train_mask_dir,
	target_size=target_size,
	batch_size=batch_size,
	class_mode=None,
	color_mode='grayscale',
	seed=1)

train_combined_generator = zip(train_generator, train_mask_generator)

validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
	validation_image_dir,
	target_size=(input_size[0], input_size[1]),  # Input size remains 1920x1080
	batch_size=batch_size,
	class_mode=None,  # Because we will use custom loss
	seed=1)

validation_mask_generator = validation_datagen.flow_from_directory(
	validation_mask_dir,
	target_size=target_size,
	batch_size=batch_size,
	class_mode=None,
	color_mode='grayscale',
	seed=1)

validation_combined_generator = zip(validation_generator, validation_mask_generator)

# Train the model
model.fit(train_combined_generator, steps_per_epoch=len(train_generator), validation_data=validation_combined_generator, validation_steps=len(validation_generator), epochs=epochs)

# Save the model
model.save('target/face_detector.keras')

print('Model trained and saved!')
