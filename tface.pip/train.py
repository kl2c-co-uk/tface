





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
if True:
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

	src_wh = input_size
	out_wh = output_size

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

	model = Model(inputs=base_model.input, outputs=x)



	# Compile the model
	model.compile(optimizer='adam', loss='mse')
	# model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

	# Display the model summary
	# model.summary()

# Paths to training and validation directories
if True:
	from dataset import dataset_main
	dataset_norms = dataset_main()
	train_image_dir = dataset_norms + '/train/images'
	train_mask_dir = dataset_norms + '/train/masks'
	validation_image_dir = dataset_norms + '/validation/images'
	validation_mask_dir = dataset_norms + '/validation/masks'

	raise Exception(
		'guess i need a new generator? https://chatgpt.com/c/e2b1d595-f434-4fad-bf80-6aa207d43f8b'
	)

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
