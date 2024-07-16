

from datasource import config


def main():

	training, validate = datasets()

	model = tface_model()

	# Compile the model
	model.compile(
		optimizer='adam',
		loss='mean_squared_error',
		metrics=['accuracy']
	)


	
	# Get the image path
	contents = 'target/cache/'
	image='de776619cedb14de4a9b6cf8f7b82265'
	raw_image = load_img(contents + image + '.jpg')

	untrained = predict(model, raw_image)
	truth =  load_img(contents + image + '.png')


	# Train the model
	history = model.fit(
		training,
		validation_data=validate,
		epochs=config.EPOCHS
	)  # Adjust the number of epoch

	trained = predict(model, raw_image)
	preview(raw_image, untrained, truth, trained)

def tface_model():
	from datasource import config


	input_shape = (config.input_height, config.input_width, 3)

	import tensorflow as tf
	from tensorflow.keras.layers import Layer, Input
	from tensorflow.keras.models import Model
	from tensorflow.keras.preprocessing import image
	import numpy as np
	import matplotlib.pyplot as plt
	from tensorflow.keras.layers import Input, Conv2D, UpSampling2D

	from tensorflow.keras.layers import Input, Resizing
	import tensorflow as tf
	import tensorflow_hub as hub
	from tensorflow.keras.layers import Input, Lambda
	from tensorflow.keras.models import Model
	from tensorflow.keras import layers, models


	# ##
	# # build a model

	input_image = Input(shape=input_shape)

	# bottom/start of the network is just ... a 1080p RGB image
	model = input_image
	

	# resize
	model = tf.keras.layers.Resizing(height=224, width=224)(model)


	# # Load the pre-trained face detection model from TensorFlow Hub
	# face_detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
	# def lamdba(images):
		
	# 	# squish into a uint8 layer
	# 	images = tf.clip_by_value(images, 0, 1)
	# 	images *= 255.0
	# 	images = tf.cast(images, tf.uint8)
		
	# 	# do the detection
	# 	images = face_detector(images)
	# 	images = images['detection_boxes']

	# 	return images
	# model = Lambda(lamdba)(model)




	# Load the ResNet50 model pre-trained on ImageNet, without the top layer
	resnet_base = tf.keras.applications.ResNet50(
		include_top=config.RESNET_TOP,
		weights='imagenet',
		input_shape=(224, 224, 3)
	)

	# Set the base model to be not trainable
	resnet_base.trainable = config.RESNET_TRAIN

	model = resnet_base(model)

	# Add global average pooling layer (because of resnet?)
	model = layers.GlobalAveragePooling2D()(model)

	model = tf.keras.layers.Flatten()(model)

	model = tf.keras.layers.Dense(config.heatmap_height * config.heatmap_width, activation='relu')(model)

	model = tf.keras.layers.Reshape((config.heatmap_height, config.heatmap_width, 1))(model)



	##
	# hekkit; just do ... dense layers?

	##
	# build it into a model
	model = Model(inputs=input_image, outputs=model)




	# this was the "old" model - might be safer
	"""
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
	"""
















	return model

def predict(model, img):
	from tensorflow.keras.preprocessing import image
	import numpy as np

	# Predict grayscale image
	grayscale_image = model.predict(
		# Expand dimensions to create a batch of size 1
		np.expand_dims(img, axis=0)
	)

	# Remove the batch dimension and squeeze the grayscale channel
	grayscale_image = np.squeeze(grayscale_image, axis=0)
	grayscale_image = np.squeeze(grayscale_image, axis=-1)
	
	return grayscale_image

def load_img(image_path):
	from tensorflow.keras.preprocessing import image
	import numpy as np

	# Load and preprocess the image
	img = image.load_img(image_path)
	img = image.img_to_array(img) / 255.0  # Normalize the image array
	
	return img

def preview(img, untrained, truth, trained):
	import matplotlib.pyplot as plt

	# Display the original and grayscale images
	plt.figure(figsize=(15, 10))  # Adjust the figure size as needed

	# First row: Original and Untrained heat map images
	plt.subplot(2, 2, 1)
	plt.title('Original RGB Image')
	plt.imshow(img)  # Display the original image

	plt.subplot(2, 2, 2)
	plt.title('Untrained Heat Map')
	plt.imshow(untrained, cmap='gray')  # Display the untrained heat map image

	# Second row: Truth heat map and Trained heat map images
	plt.subplot(2, 2, 3)
	plt.title('Truth Heat Map')
	plt.imshow(truth, cmap='gray')  # Display the truth heat map image

	plt.subplot(2, 2, 4)
	plt.title('Predicted Heat Map')
	plt.imshow(trained, cmap='gray')  # Display the trained heat map image

	plt.show()

def datasets():
	"""
	return training and validation dataset objects
	"""
	
	import tensorflow as tf
	
	def edge(yset):



		# Convert the list of pairs to a TensorFlow Dataset
		dataset = map(lambda d: (d.cache[0], d.cache[1]), yset)
		dataset = list(dataset)
		dataset = tf.data.Dataset.from_tensor_slices(dataset)

		def preprocess(v):

			input_path, target_path = tf.unstack(v)

			input_image = tf.io.read_file(input_path)
			input_image = tf.image.decode_jpeg(input_image, channels=3)  # Ensure RGB
			input_image = tf.image.convert_image_dtype(input_image, tf.float32)  # Convert to float32
			input_image = input_image / 255.0  # Normalize to [0, 1]

			target_image = tf.io.read_file(target_path)
			target_image = tf.image.decode_png(target_image, channels=1)  # Ensure grayscale
			target_image = tf.image.convert_image_dtype(target_image, tf.float32)  # Convert to float32
			target_image = target_image / 255.0  # Normalize to [0, 1]

			return input_image, target_image

		# Map the dataset using the unpack_and_preprocess function
		dataset = dataset.map(
			preprocess,
			num_parallel_calls=tf.data.AUTOTUNE
		)

		# finish it
		dataset = dataset.batch(config.BATCH_SIZE)
		dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

		return dataset

	from datasource.base import Cache 
	import dataset
	cache = Cache('target/')

	t = edge(dataset.yset_training(cache))
	v = edge(dataset.yset_validate(cache))

	return (t, v)

if __name__ == '__main__':
	main()
