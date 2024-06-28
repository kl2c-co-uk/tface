"""

 λ nodemon --ignore target/ dataset.p

 """

target_width = 1920
target_height = 1080

import os
import requests
import zipfile
import cv2

def build_dataset():
	"""build the dataset

	build a dataset of many images resized and as heat-maps
	"""

	download_file(
		'https://drive.usercontent.google.com/download?id=15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M&export=download&authuser=0&confirm=t&uuid=6d1b1482-0707-4fee-aca1-0ea41ba1ecb6&at=APZUnTX8U1BtsQRxJTqGH5qAbkFf%3A1719226478335',
		'target/wider.training.zip'
	)
	download_file(
		'https://drive.usercontent.google.com/download?id=1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q&export=download&authuser=0&confirm=t&uuid=8afa3062-ddbc-44e5-83fd-c4e1e2965513&at=APZUnTUX4c1Le0kpmfMNJ6i3cIJh%3A1719227725353',
		'target/wider.validation.zip'
	)
	download_file(
		'http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip',
		'target/wider.annotations.zip'
	)

	# we start with the annotations file (sorry)
	for image, faces in wider_faces('wider_face_train_bbx_gt.txt'):
		for _, data in zipfile_get('target/wider.training.zip', lambda img: img.endswith(image)):

			# repack image
			(faces, data) = repack_image(faces, data, target_width, target_height)

			throw('save repacked image?')

	throw('??? - that seems to be the training images?')

def wider_faces(labels):
	# we start with the annotations file (sorry)
	for _, text in zipfile_get('target/wider.annotations.zip', lambda get: get.endswith(labels)):
		
		# turn it into a more conventional iterator
		text = literator(text.decode('utf-8').splitlines())

		# decode each entry
		while text.more():

			# we NEED to decode each entry from the iterator (even if we don't need to decompress the file)
			image = text.take()
			count = int(text.take())
			faces = []
			if 0 == count:
				# for extra weirdness; entries (or The One Entry) with no faces have a line with garbage data
				blank = text.take().strip()
				if '0 0 0 0 0 0 0 0 0 0 '.strip() != blank:
					throw('empty entry had a funky line!')
			else:
				while len(faces) < count:
					x, y, w, h, *_ = text.take().split(' ')
					faces.append((x, y, w, h))
			
			# we don't need to mess with them here
			yield (image, faces)

def zipfile_get(file, test):
	found = False

	for item in zipfile_all(file, test):
		if found:
			throw('too many entries match')
		else:
			yield item
	if not found:
		throw('no entry matched')



def zipfile_all(file, test):
	with zipfile.ZipFile(file, 'r') as file:
		for info in file.infolist():
			name = info.filename
			if test(name):
				yield (name, file.read(info))


class literator():
	def __init__(self, list):
		self._next = 0
		self._list = list
	
	def more(self):
		return self._next < len(self._list)
	
	def take(self):
		item = self._list[self._next]
		self._next += 1
		return item

from PIL import Image
from io import BytesIO


def repack_image(faces, data, target_width, target_height):
	import random
	target_image = Image.new('RGB', (target_width, target_height), tuple(random.randint(0, 255) for _ in range(3)))

	image = Image.open(BytesIO(data))

	(width, height) = image.size
	
	scaled_w = '?setme?'
	scaled_h = '?setme?'
	scaled_x = '?setme?'
	scaled_y = '?setme?'

	print(f'(width, height) = {(width, height)}')

	if width == target_width and height == target_height:
		throw('no need to scale image!')
	elif width < height:

		# compute the scaled width
		scaled_w = int((width * target_height) / height)
		
		scaled_h = target_height
		scaled_y = 0

		assert scaled_w < target_width

		scaled_x = random.randint(0, (target_width - scaled_w))
	else:
		throw('were wide')

	# scale the image
	scaled_image = image.resize((scaled_w, scaled_h), Image.LANCZOS)

	# now overlay image
	target_image.paste(scaled_image, (scaled_x, scaled_y))

	throw('??? - now scale faces')


	throw('??? - pack and return this')

	








def download_file(url, save_path):

	if None == save_path:
		name = 'target/' + md5(url)
		download_file(url, name)
		return name

	
	ensure_directory_exists(save_path)

	if os.path.exists(save_path):
			# print(f"The file '{save_path}' already exists. Skipping download.")
			return

	print('this could be loonnngggg .....')
	
	response = requests.get(url, stream=True)
	with open(save_path, 'wb') as f:
			for chunk in response.iter_content(chunk_size=8192):
					if chunk:
							f.write(chunk)
	
	print(f"Downloaded {url} to {save_path}")

import hashlib

def md5(input_string):
    # Encode the string to bytes, then create an MD5 hash object
    md5_hash = hashlib.md5(input_string.encode())

    # Get the hexadecimal representation of the hash
    return md5_hash.hexdigest()



def ensure_directory_exists(file_path):
	directory = os.path.dirname(file_path)
	if not os.path.exists(directory):
		os.makedirs(directory)
	return directory














##
# wider! http://shuoyang1213.me/WIDERFACE/
# if this doesn't work, visit the website
download_file(
	'https://drive.usercontent.google.com/download?id=15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M&export=download&authuser=0&confirm=t&uuid=6d1b1482-0707-4fee-aca1-0ea41ba1ecb6&at=APZUnTX8U1BtsQRxJTqGH5qAbkFf%3A1719226478335',
	'target/wider.training.zip'
)

download_file(
	'https://drive.usercontent.google.com/download?id=1HIfDbVEWKmsYKJZm4lchTBDLW5N7dY5T&export=download&authuser=0&confirm=t&uuid=7afbbdc2-cbaf-4d4a-8998-16296c6d7ccd&at=APZUnTVUADxFyK6lmy5VgFTEYfUy%3A1719227735059',
	'target/wider.testing.zip'
)

download_file(
	'https://drive.usercontent.google.com/download?id=1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q&export=download&authuser=0&confirm=t&uuid=8afa3062-ddbc-44e5-83fd-c4e1e2965513&at=APZUnTUX4c1Le0kpmfMNJ6i3cIJh%3A1719227725353',
	'target/wider.validation.zip'
)

download_file(
	'http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip',
	'target/wider.annotations.zip'
)

def safe_write(path):
	ensure_directory_exists(path)
	return open(path, 'w')


def throw(message):
	print('\n')
	raise Exception(f"{message}\n\n")

def wider_set(root, images, labels):
	# extract the images
	with zipfile.ZipFile(images, 'r') as full:
		for file_info in full.infolist():
			if file_info.filename.endswith('.jpg'):
				name = file_info.filename 
				name = root+name[name.find('images/'):]
				if not os.path.isfile(name):
					ensure_directory_exists(name)
					with open(name, 'wb') as file:
						file.write(full.read(file_info))

	# extract the labels
	with zipfile.ZipFile('target/wider.annotations.zip', 'r').open(labels) as file_in_zip:
		
		##
		# finky recreation of an iterator
		data = {
			'next': 0,
			'full': file_in_zip.read().decode('utf-8').splitlines(),
			'more': True
		}
		
		def line(data):
			item = data['full'][data['next']]
			data['next'] += 1
			data['more'] = data['next'] < len(data['full'])
			return item

		
		print('')

		
		def xml_write(xml, string):
			xml.write(string + '\n')

		while data['more']:

			# read the file name for this entry
			name = line(data)
			if not name.endswith('.jpg'):
				throw(f"the file `{name}` does not end with .jpg(next = {data['next']})")

			# print(f'doing {name} annotation ...')

			# open teh label xml file
			xml = root+ 'annotations/' +name[:-3] + 'xml'
			if not os.path.isfile(xml):
				xml = safe_write(xml)
			else:
				xml = False

			# load the count of targets
			count = int(line(data))

			if xml:
				# load the image and find its dimensions
				image = root+'images/'+name
				if not os.path.isfile(image):
					print('\n')
					throw(f"the file `{image}` does not exist (next = {data['next']})")
				image = cv2.imread(image)
				if image is None:
					throw(f'failed to load {name}')
				height, width, channels = image.shape

				# write the header for the record
				xml_write(xml, f'<annotation>')
				xml_write(xml, f'\t<filename>{name}</filename>')
				xml_write(xml, f'\t<size>')
				xml_write(xml, f'\t\t<width>{width}</width>')
				xml_write(xml, f'\t\t<height>{height}</height>')
				xml_write(xml, f'\t</size>')

			# wander through each item in the record
			if 0 == count:
				blank = line(data).strip()
				if '0 0 0 0 0 0 0 0 0 0 '.strip() != blank:
					throw('empty entry had a funky line!')
			else:
				for i in range(count):
					text = line(data).split(' ')
					x, y, w, h, *_ = text

					if xml:
						xml_write(xml, f'\t<object>')
						xml_write(xml, f'\t\t<name>face</name>')
						xml_write(xml, f'\t\t<bndbox>')
						xml_write(xml, f'\t\t\t<xmin>{x}</xmin>')
						xml_write(xml, f'\t\t\t<ymin>{y}</ymin>')
						xml_write(xml, f'\t\t\t<xmax>{x+w}</xmax>')
						xml_write(xml, f'\t\t\t<ymax>{y+h}</ymax>')
						xml_write(xml, f'\t\t</bndbox>')
						xml_write(xml, f'\t</object>')

			if xml:
				xml_write(xml, f'</annotation>')
				xml.close()

if __name__ == "__main__":
    build_dataset()
else:
	throw('do something else?')

	# the validation set
	wider_set(
		root = 'target/dataset/validation/',
		images = 'target/wider.validation.zip',
		labels = 'wider_face_split/wider_face_val_bbx_gt.txt'
	)

	# the training set
	wider_set(
		root = 'target/dataset/train/',
		images = 'target/wider.training.zip',
		labels = 'wider_face_split/wider_face_train_bbx_gt.txt'
	)


print("data set loaded - okie dokee")


