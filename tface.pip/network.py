





# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import Resizing
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D
from tensorflow.keras.models import Model

# Define the model for 1920x1080 input and 192x108 output
def face_detector(src_wh, out_wh):

	# Define input shape
	input_shape = (src_wh[0], src_wh[1], 3)  # Adjust based on resizing

	# Load ResNet50 with pretrained weights, exclude the top layers
	base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

	# Build custom head for heat map prediction
	x = base_model.output
	x = UpSampling2D(size=(6, 6))(x)  # Adjust based on desired output size
	x = Conv2D(out_wh[0], (3, 3), activation='relu', padding='same')(x)
	x = Conv2D(out_wh[1], (3, 3), activation='relu', padding='same')(x)
	x = Conv2D(1, (1, 1), activation='sigmoid')(x)  # Assuming single channel heat map

	# Define the model
	model = Model(inputs=base_model.input, outputs=x)
	return model

	# # Compile the model
	# model.compile(optimizer='adam', loss='mse')

	# # Display the model summary
	# model.summary()



