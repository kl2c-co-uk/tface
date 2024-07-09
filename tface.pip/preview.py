import tensorflow as tf
from tensorflow.keras.layers import Layer, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D


import dataset


def main():
	input_shape = (1080, 1920, 3)

	##
	# build a model

	input_image = Input(shape=input_shape)

	# bottom/start of the network is just ... a 1080p RGB image
	model = input_image

	# convert it to greyscale
	model = RGBToGrayscaleLayer()(model)


	model = Model(inputs=input_image, outputs=model)


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

# Custom layer to convert RGB image to grayscale using specified weights
class RGBToGrayscaleLayer(Layer):
    def __init__(self, **kwargs):
        super(RGBToGrayscaleLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Extract RGB channels
        r, g, b = inputs[..., 0], inputs[..., 1], inputs[..., 2]

        # Convert to grayscale using specified weights
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        
        # Expand dimensions to make it (batch_size, height, width, 1)
        gray = tf.expand_dims(gray, axis=-1)
        
        return gray


if __name__ == '__main__':
	main()
