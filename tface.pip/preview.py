
DEBUG_PRINTS = False

def main():

	training, validate = datasets(batch_size=4)

	model = tface_model()

	preview(model)

	# Compile the model
	model.compile(optimizer='adam',
		loss='mean_squared_error',
		metrics=['accuracy'])

	# Train the model
	history = model.fit(training,
			validation_data=validate,
			epochs=10)  # Adjust the number of epoch

	preview(model)

def tface_model():
	input_width = 1920
	input_height = 1080
	heat_width = 192
	heat_height = 108

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


	# ##
	# # build a model

	input_image = Input(shape=input_shape)

	# bottom/start of the network is just ... a 1080p RGB image
	model = input_image
	
	if DEBUG_PRINTS:print(f'''\n>>>input
	{model}
	''')

	# resize
	model = tf.keras.layers.Resizing(height=224, width=224)(model)


	if DEBUG_PRINTS:print(f'''\n>>>resized
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


	if DEBUG_PRINTS:print(f'''\n>>>face_detector
	{model}
	''')


	##
	#


	# gpt says; Assume detection_boxes has shape [num_detections, 4]
	# pal says; ... WtH?
	num_detections = tf.shape(model)[0]

	if DEBUG_PRINTS:print('got one thing')

	# set our dense layer thing

	model = tf.keras.layers.Flatten()(model)

	if DEBUG_PRINTS:print('did i flatten?')

	model = tf.keras.layers.Dense(heat_height * heat_width, activation='relu')(model)

	if DEBUG_PRINTS:print('did i dense?')
	
	model = tf.keras.layers.Reshape((heat_height, heat_width, 1))(model)

	if DEBUG_PRINTS:print('reshaped!')


	##
	# hekkit; just do ... dense layers?

	##
	# build it into a model
	model = Model(inputs=input_image, outputs=model)

	##
	# show some junk about the model
	if DEBUG_PRINTS:model.summary()

	return model

def preview(model):

	##
	# run an image (of emma watson?) through the model

	from dataset import contents
	image_path = contents()[0] + '/de776619cedb14de4a9b6cf8f7b82265.jpg'

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



	img = image.load_img(image_path, target_size=(1080, 1920))
	img = image.img_to_array(img)

	# Normalize the image array
	img = img / 255.0

	# Expand dimensions to create a batch of size 1
	img = np.expand_dims(img, axis=0)

	# Predict grayscale image
	grayscale_image = model.predict(img)

	if DEBUG_PRINTS:print("=="*10)

	if DEBUG_PRINTS:print(grayscale_image)


	# Remove the batch dimension and squeeze the grayscale channel
	grayscale_image = np.squeeze(grayscale_image, axis=0)
	grayscale_image = np.squeeze(grayscale_image, axis=-1)

	# switch the image back to being just one
	img = img[0]

	# check that it's what we expect
	if DEBUG_PRINTS:print(
		f"""=====

		len(img) = {len(img)}
		len(img[0]) = {len(img[0])}
		len(img[0][0]) = ({len(img[0][0])})
		type(img[0][0]) = ({type(img[0][0])})

		len(grayscale_image) = {len(grayscale_image)}
		len(grayscale_image[0]) = {len(grayscale_image[0])}
		len(grayscale_image[0][0]) =(No!)
		type(grayscale_image[0][0]) ={type(grayscale_image[0][0])}

		"""
	)
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

def datasets(batch_size):

	# train_image_dir, train_mask_dir, validation_image_dir, validation_mask_dir = contents()

	# print(train_image_dir)
	# print(train_mask_dir)

	# root = dataset_main()
	root = 'target/mega-wipder-data/'

	import os

	src_width = 1920
	src_height = 1080
	out_width = 192
	out_height = 108

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
