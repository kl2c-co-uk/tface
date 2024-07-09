import tensorflow as tf
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D


import dataset

class RGBToGrayscaleLayer(Layer):
	def __init__(self, **kwargs):
		super(RGBToGrayscaleLayer, self).__init__(**kwargs)

	def call(self, inputs):
		r, g, b = inputs[..., 0], inputs[..., 1], inputs[..., 2]
		gray = 0.299 * r + 0.587 * g + 0.114 * b
		return tf.expand_dims(gray, axis=-1)

if __name__ == '__main__':
	print('hey dude')

	input_shape = (1080, 1920, 3)

	input_image = Input(shape=input_shape)
	model = input_image
	model = RGBToGrayscaleLayer()(model)

	# Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation=None)




	model = Model(inputs=input_image, outputs=model)

	model.summary()

	image_path, _, _, _ = dataset.contents()

	image_path += '/de776619cedb14de4a9b6cf8f7b82265.jpg'

	img = image.load_img(image_path, target_size=(1080, 1920))
	img_array = image.img_to_array(img)

	# Normalize the image array
	img_array = img_array / 255.0

	# Expand dimensions to create a batch of size 1
	img_array = np.expand_dims(img_array, axis=0)

	# Predict grayscale image
	grayscale_image = model.predict(img_array)

	# Remove the batch dimension and squeeze the grayscale channel
	grayscale_image = np.squeeze(grayscale_image, axis=0)
	grayscale_image = np.squeeze(grayscale_image, axis=-1)

	# Display the original and grayscale images
	plt.figure(figsize=(10, 5))

	plt.subplot(1, 2, 1)
	plt.title('Original RGB Image')
	plt.imshow(img_array[0])  # Display the original image

	plt.subplot(1, 2, 2)
	plt.title('Grayscale Image')
	plt.imshow(grayscale_image, cmap='gray')  # Display the grayscale image

	plt.show()
