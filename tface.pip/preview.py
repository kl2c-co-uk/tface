import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import matplotlib.pyplot as plt

def preprocess_image(image_path):
	# Load the image
	img = cv2.imread(image_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img = cv2.resize(img, (224, 224))
	img = np.expand_dims(img, axis=0)
	img = img / 255.0  # Normalize to [0, 1] range
	return img

def segment_face(model, image_path):
	# Preprocess the image
	img = preprocess_image(image_path)

	# Run the model
	predictions = model(img)
	predictions = predictions["semantic_pred"]

	# Ensure predictions are not empty
	if predictions.size == 0:
		raise ValueError("Model prediction returned an empty result.")

	# Post-process the output
	mask = predictions[0].numpy()  # Convert TensorFlow tensor to numpy array
	mask = (mask == 15)  # Assuming class 15 corresponds to 'person' (varies by model)

	# Resize the mask to the original image size
	original_img = cv2.imread(image_path)
	original_size = (original_img.shape[1], original_img.shape[0])
	mask = cv2.resize(mask.astype(np.uint8), original_size)

	return original_img, mask
 
def visualize_segmentation(image, mask):
	plt.figure(figsize=(10, 10))
	plt.subplot(1, 2, 1)
	plt.title("Original Image")
	plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	plt.axis('off')

	plt.subplot(1, 2, 2)
	plt.title("Segmented Face")
	plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
	plt.imshow(mask, alpha=0.5, cmap='jet')
	plt.axis('off')
	plt.show()

if '__main__' == __name__:
	print('hey dude')

	# Load the pre-trained DeepLabV3 model from TensorFlow Hub
	MODEL_URL = "https://tfhub.dev/tensorflow/deeplabv3/1/deeplabv3_mnv2_pascal_train_aug/1"
	MODEL_URL = 'https://www.kaggle.com/models/tensorflow/deeplab/tfjs/pascal/1/model.json?tfjs-format=file'
	model = hub.load(MODEL_URL)



	# Example usage
	image_path = 'path_to_your_image.jpg'
	image_path = 'target/square.jpg'
	image, mask = segment_face(model, image_path)
	visualize_segmentation(image, mask)
