
EPOCHS = 1




def main():

	training, validate = datasets(batch_size=4)

	model = tface_model()

	# Compile the model
	model.compile(optimizer='adam',
		loss='mean_squared_error',
		metrics=['accuracy'])

	
	# Get the image path
	from dataset import contents
	image='de776619cedb14de4a9b6cf8f7b82265'
	raw_image = load_img(contents()[0] + '/' + image + '.jpg')

	untrained = predict(model, raw_image)
	truth =  load_img(contents()[1] + '/' + image + '.png')


	# Train the model
	history = model.fit(
		training,
		validation_data=validate,
		epochs=EPOCHS
	)  # Adjust the number of epoch

	trained = predict(model, raw_image)
	preview(raw_image, untrained, truth, trained)

def tface_model():
	import dataset
	input_width, input_height, heat_width, heat_height = dataset.sizes()

	input_shape = (input_height, input_width, 3)

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
	resnet_base = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

	# Set the base model to be not trainable
	# resnet_base.trainable = False

	model = resnet_base(model)

	# Add global average pooling layer (because of resnet?)
	model = layers.GlobalAveragePooling2D()(model)

	model = tf.keras.layers.Flatten()(model)

	model = tf.keras.layers.Dense(heat_height * heat_width, activation='relu')(model)

	model = tf.keras.layers.Reshape((heat_height, heat_width, 1))(model)



	##
	# hekkit; just do ... dense layers?

	##
	# build it into a model
	model = Model(inputs=input_image, outputs=model)

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
	plt.title('Trained Heat Map')
	plt.imshow(trained, cmap='gray')  # Display the trained heat map image

	plt.show()

def datasets(batch_size):

	from dataset import contents
	train_image_dir, train_mask_dir, validation_image_dir, validation_mask_dir = contents()

	# print(train_image_dir)
	# print(train_mask_dir)

	# root = dataset_main()
	root = 'target/mega-wipder-data/'

	import os, dataset

	src_width, src_height, out_width, out_height = dataset.sizes()

	def mask_set(root):
		import tensorflow as tf
		im = 'images/'
		ma = 'masks/'
		for s in os.listdir(root + im):
			assert s.endswith('jpg')
			o = s[:-3]+'png'
			assert os.path.isfile(root + ma + o)

		srcs = tf.data.Dataset.list_files(os.path.join(root + im, '*.jpg'))
		outs = tf.data.Dataset.list_files(os.path.join(root + ma, '*.png'))

		# i really want to doublecheck these - but - ugghhh

		def prep_src(src):
			src = tf.io.read_file(src)
			src = tf.image.decode_jpeg(src, channels=3)
			src = tf.image.resize(src, [src_height, src_width])
			src = tf.cast(src, tf.float32) / 255.0
			return src

		def prep_out(out):
			out = tf.io.read_file(out)
			out = tf.image.decode_png(out, channels=1)
			out = tf.image.resize(out, [out_height, out_width])
			out = tf.cast(out, tf.float32) / 255.0
			return out
		
		dataset = tf.data.Dataset.zip((srcs, outs))
		dataset = dataset.map(lambda x, y: (prep_src(x), prep_out(y)),num_parallel_calls=tf.data.AUTOTUNE)
		dataset = dataset.batch(batch_size)
		dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
		return dataset

	training = mask_set(root + 'train/')
	validate = mask_set(root + 'validation/')

	return (training, validate)

if __name__ == '__main__':
	main()
