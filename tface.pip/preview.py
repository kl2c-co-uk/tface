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

import dataset

def modelmake_model(input_shape):
	import tensorflow as tf
	import tensorflow_hub as hub
	from tensorflow.keras.layers import Input, Lambda
	from tensorflow.keras.models import Model

	# Load the pre-trained face detection model from TensorFlow Hub
	face_detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
	detector_output = face_detector(images)
	return detector_output['detection_boxes']

	def detect_faces(images):
		return face_detector(images)['detection_boxes']

	# Assuming input_shape is defined somewhere before
	input_image = Input(shape=input_shape)

	# Resize the input image to a desired size, e.g., (224, 224)
	resized_image = tf.keras.layers.Resizing(height=224, width=224)(input_image)

	# Convert resized image to grayscale
	grayscale_image = tf.image.rgb_to_grayscale(resized_image)

	# Detect faces in the image
	

	# Combine grayscale image with face detection boxes for visualization or further processing
	combined_output = Lambda(lambda x: (x[0], x[1]))([grayscale_image, faces])

	# Define the model
	model = Model(inputs=input_image, outputs=combined_output)

	# Model summary
	model.summary()
	return model

def main():
	input_shape = (1080, 1920, 3)

	# ##
	# # build a model

	input_image = Input(shape=input_shape)

	# bottom/start of the network is just ... a 1080p RGB image
	model = input_image
	
	print(f'''\n>>>input
	{model}
	''')

	# convert it to greyscale
	# model = tf.image.rgb_to_grayscale(model)
	# print(f'''>>>	greyscale
	# {model}
	# ''')


	# resize
	model = tf.keras.layers.Resizing(height=224, width=224)(model)


	print(f'''\n>>>resized
	{model}
	''')



	# Load the pre-trained face detection model from TensorFlow Hub
	face_detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
	def lamdba(images):
		
		# squish into a uint8 layer
		images = tf.clip_by_value(images, 0, 1)
		images *= 255.0
		images = tf.cast(images, tf.uint8)
		
		# do the detection
		images = face_detector(images)
		images = images['detection_boxes']
		return images
	model = Lambda(lamdba)(model)


	print(f'''\n>>>face_detector
	{model}
	''')


	raise Exception(
		'so i think we now have 1x100x4 tensor with the deteacted faces in some form coming out of it'
	)



	model = Model(inputs=input_image, outputs=model)


	# raise '???'

	##
	# show some junk about the model
	model.summary()





	##
	# run an image (of emma watson?) through the model

	image_path = dataset.contents()[0] + '/de776619cedb14de4a9b6cf8f7b82265.jpg'

	img = image.load_img(image_path, target_size=(1080, 1920))
	img = image.img_to_array(img)

	# Normalize the image array
	img = img / 255.0

	# Expand dimensions to create a batch of size 1
	img = np.expand_dims(img, axis=0)

	# Predict grayscale image
	grayscale_image = model.predict(img)

	# Remove the batch dimension and squeeze the grayscale channel
	grayscale_image = np.squeeze(grayscale_image, axis=0)
	grayscale_image = np.squeeze(grayscale_image, axis=-1)

	# switch the image back to being just one
	img = img[0]

	# check that it's what we expect
	# print(
	# 	f"""=====

	# 	len(img) = {len(img)}
	# 	len(img[0]) = {len(img[0])}
	# 	len(img[0][0]) = ({len(img[0][0])})
	# 	type(img[0][0]) = ({type(img[0][0])})

	# 	len(grayscale_image) = {len(grayscale_image)}
	# 	len(grayscale_image[0]) = {len(grayscale_image[0])}
	# 	len(grayscale_image[0][0]) =(No!)
	# 	type(grayscale_image[0][0]) ={type(grayscale_image[0][0])}

	# 	"""
	# )
	assert "<class 'numpy.float32'>" ==str(type(grayscale_image[0][0]))


	##
	# show what the thingie has made fromt hat image

	# Display the original and grayscale images
	plt.figure(figsize=(10, 5))

	plt.subplot(1, 2, 1)
	plt.title('Original RGB Image')
	plt.imshow(img)  # Display the original image

	plt.subplot(1, 2, 2)
	plt.title('Grayscale Image')
	plt.imshow(grayscale_image, cmap='gray')  # Display the grayscale image)  # Display the grayscale image

	plt.show()



if __name__ == '__main__':
	main()
