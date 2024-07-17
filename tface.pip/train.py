

from datasource import config
from datasource import Cache 


from enum import Enum
class Mode(Enum):
	HEAT_MAP = 1
	PATCHES = 2

mode = Mode.PATCHES

def main():

	training, validate = datasets()

	model = tface_model()
	
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


	input_shape = (config.INPUT_HEIGHT, config.INPUT_WIDTH, 3)

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


	# Load the ResNet50 model pre-trained on ImageNet, without the top layer
	resnet_base = tf.keras.applications.ResNet50(
		include_top=config.RESNET_TOP,
		weights='imagenet',
		input_shape=input_shape
	)
	resnet_base.trainable = config.RESNET_TRAIN

	model = resnet_base(model)



	if Mode.HEAT_MAP == mode:
		# Add global average pooling layer (because of resnet?)
		model = layers.GlobalAveragePooling2D()(model)

		model = tf.keras.layers.Flatten()(model)

		# the "heat maps" model
		model = tf.keras.layers.Dense(config.HEATMAP_HEIGHT * config.HEATMAP_WIDTH, activation='relu')(model)
		model = tf.keras.layers.Reshape((config.HEATMAP_HEIGHT, config.HEATMAP_WIDTH, 1))(model)

		# build it into a model
		model = Model(inputs=input_image, outputs=model)

		# Compile the model
		model.compile(
			optimizer='adam',
			loss='mean_squared_error',
			# loss='binary_crossentropy',
			metrics=['accuracy']
		)
		
		# we be done
		return model

	elif Mode.PATCHES == mode:
		import tensorflow.keras.layers as layer


		# Define the RPN (Region Proposal Network)
		rpn = layer.Conv2D(512, (3, 3), padding='same', activation='relu')(model)
		rpn_class = layer.Conv2D(config.PATCH_COUNT, (1, 1), activation='sigmoid')(rpn)  # config.PATCH_COUNT anchor boxes
		rpn_bbox = layer.Conv2D(config.PATCH_COUNT, (1, 1))(rpn)  # 4 coordinates for each of the config.PATCH_COUNT anchor boxes

		# Flatten the RPN outputs
		rpn_class_flat = layer.Flatten()(rpn_class)
		rpn_bbox_flat = layer.Flatten()(rpn_bbox)

		# "anchors?"
		# # # Define the RPN model
		# # rpn_model = Model(inputs=input_image, outputs=[rpn_class_flat, rpn_bbox_flat])

		# # # Print the model summary
		# # rpn_model.summary()

		# # # PAL SAYS;
		# # # i asmed about the ancor boxes, but, the answer eluded me https://chatgpt.com/c/bfb8472a-bb2e-4b9e-b128-a2d5e4d0e7d0

		# ... trying to BASH it into shape with big dumb ones
		rpn_both = layer.Concatenate()([rpn_class_flat, rpn_bbox_flat])

		width = config.PATCH_COUNT * 4
		rpn_thin = layer.Dense(width * 4, activation='relu')(rpn_both)
		rpn_thin = layer.Dense(width, activation='relu')(rpn_thin)

		rpn_dumb = Model(inputs=input_image, outputs=rpn_thin)
		rpn_dumb.summary()

		
		# Compile the model
		rpn_dumb.compile(
			optimizer='adam',
			loss='mean_squared_error',
			# loss='binary_crossentropy',
			metrics=['accuracy']
		)
		
		# we be done
		return rpn_dumb
		
	else:
		raise Exception('??? mode = {mode}')


	# this was the "old" model - might be something to expolore
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

	if Mode.HEAT_MAP == mode:
		# Predict grayscale image
		grayscale_image = model.predict(
			# Expand dimensions to create a batch of size 1
			np.expand_dims(img, axis=0)
		)

		# Remove the batch dimension and squeeze the grayscale channel
		grayscale_image = np.squeeze(grayscale_image, axis=0)
		grayscale_image = np.squeeze(grayscale_image, axis=-1)
		
		return grayscale_image
	elif Mode.PATCHES == mode:

		predict = model.predict(
			# Expand dimensions to create a batch of size 1
			np.expand_dims(img, axis=0)
		)
		
		# the thing. the list-of-points
		raise Exception(f'decode the patches from predict >>>{predict}<<<')

		
		raise Exception('create a blank')
		raise Exception('fill in the patches in a blank')
		raise Exception('return it!')
		
	else:
		raise Exception('??? mode = {mode}')


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
	
	edge = '?>' # lambda a : raise Exception('set the edge function')

	def preprocess_jpeg(path):
		
		channels = 3

		image = tf.io.read_file(path)
		image = tf.image.decode_jpeg(image, channels=channels)  # Ensure RGB
		image = tf.image.convert_image_dtype(image, tf.float32)  # Convert to float32
		image = image / 255.0  # Normalize to [0, 1]

		return image

	if Mode.HEAT_MAP == mode:
		def heat_maps(yset):

			# Convert the list of pairs to a TensorFlow Dataset
			dataset = map(lambda d: (d.cache[0], d.cache[1]), yset)
			dataset = list(dataset)
			dataset = tf.data.Dataset.from_tensor_slices(dataset)

			def preprocess(v):

				input_path, target_path = tf.unstack(v)

				input_image = preprocess_jpeg(input_path)

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
		edge = heat_maps
	elif Mode.PATCHES == mode:
		def patches(yset):

			# Convert the list of tuples to a TensorFlow Dataset of pairs
			dataset = map(lambda d: (d.cache[0], d.cache[2]), yset)
			dataset = list(dataset)
			dataset = tf.data.Dataset.from_tensor_slices(dataset)
			def preprocess(v):

				input_image, target_path = tf.unstack(v)

				# read the jpeg as normal
				input_image = preprocess_jpeg(input_image)

				# read the json ... less normally
				target_patches = tf.io.read_file(target_path)

				

				def parse_bounding_boxes(json_string):
					# data = json.loads(json_string.numpy())
					# return tf.convert_to_tensor(data, dtype=tf.float32)
					raise Exception(
					f"""
						type(json_string) = {type(json_string)}
						dir(json_string) = {dir(json_string)}

						json_string = {json_string}

						
					""")
				target_patches = tf.py_function(parse_bounding_boxes, [target_patches], tf.float32)
				target_patches.set_shape([config.PATCH_COUNT, 4])
				
				return input_image, target_patches


			# Map the dataset using the unpack_and_preprocess function
			dataset = dataset.map(
				preprocess,
				num_parallel_calls=tf.data.AUTOTUNE
			)

			# finish it
			dataset = dataset.batch(config.BATCH_SIZE)
			dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

			return dataset

		edge = patches		
	else:
		raise Exception('??? mode = {mode}')


	import dataset
	cache = Cache('target/')

	t = edge(dataset.yset_training(cache))
	v = edge(dataset.yset_validate(cache))

	return (t, v)

if __name__ == '__main__':
	main()
