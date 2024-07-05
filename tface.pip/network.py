





# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import Resizing

# Define the model for 1920x1080 input and 192x108 output
def main(input_size):
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


