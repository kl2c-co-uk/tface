


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import Resizing




import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

def preprocess_image(image_path):
	# Load the image
	img = cv2.imread(image_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (224, 224))
	img = np.expand_dims(img, axis=0)
	img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
	return img

def segment_face(image_path):
	# Preprocess the image
	img = preprocess_image(image_path)

	# Run the model
	predictions = model.predict(img)

	# Ensure predictions are not empty
	if predictions.size == 0:
		raise ValueError("Model prediction returned an empty result.")

	# Post-process the output
	# For simplicity, let's assume the output is a mask
	# (in practice, you'd need to handle the specific output format of your model)
	mask = predictions[0] > 0.5  # Example thresholding

	# mask = predictions[0] #> 0.5  # Example thresholding

	raise Exception('mask = ' + str(len(mask)))


	# Resize the mask to the original image size
	original_img = cv2.imread(image_path)
	original_size = (original_img.shape[1], original_img.shape[0])
	mask = cv2.resize(mask.astype(np.uint8), original_size)

	return original_img, mask

def visualize_segmentation(image, mask):
	plt.figure(figsize=(10, 10))
	plt.subplot(1, 2, 1)
	plt.title("Original Image")
	plt.imshow(image)
	plt.axis('off')

	plt.subplot(1, 2, 2)
	plt.title("Segmented Face")
	plt.imshow(image)
	plt.imshow(mask, alpha=0.5, cmap='jet')
	plt.axis('off')
	plt.show()


if '__main__' == __name__:
	print('hey dude')
	import dataset
	train_image_dir, train_mask_dir, validation_image_dir, validation_mask_dir =  dataset.contents()
	# Load the pre-trained DeepLabV3 model from TensorFlow Hub
	model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)

	# Example usage
	image_path = 'path_to_your_image.jpg'
	image_path = 'target/square.jpg'
	image, mask = segment_face(image_path)
	visualize_segmentation(image, mask)






