
from datasource import val

class FacePatch:
	def __init__(self, **kwargs):
		# collect the key/vals
		data = {}	
		for key, value in kwargs.items():
			data[key] = value


		if 1 == len(data.keys()) and 'ltrb' in data.keys():
			data = data['ltrb']

			if type([]) == type(data):
				self.l, self.t, self.r, self.b = map(int, data)
				return
		
		raise Exception('error; unhandled case for face pathc')
	
	def __str__(self):
		txt = 'FacePatch{'
		for key, value in self.__dict__.items():
			txt += str(key) + ":" + str(value) + ","
		return txt + "}"
	
	def __repr__(self):
		return self.__str__()

class DataPoint:
	def __init__(self, path, patches):
		self.path = path
		self.patches = patches
		
	def __str__(self):
		txt = 'DataPoint{'
		for key, value in self.__dict__.items():
			txt += str(key) + ":" + str(value) + ","
		return txt + "}"
	def __repr__(self):
		return self.__str__()
	
	@val
	def fKey(self):
		return md5(self.path)


from datasource import Blurb, Cache, md5, ZipWalk, ensure_directory_exists, random_split, take
import datasource.config as config
from PIL import Image


def split_export(datapoints, train, val, archive):

	# split them train:val
	split = random_split(datapoints, train, val)
	
	# limit ourselves (for testing)
	todo = take(split, config.LIMIT) if config.LIMIT > 0 else split

	# conver tot a key/value thingie to speed up lookups
	want = {}
	for item in todo:
		name = (item[0] if not item[1] else item[1]).path
		assert name not in want
		assert name == (name.strip())
		want[name] = item
	
	# scan through the zip file, and, for each item loop up of there's a deatpoint
	import zipfile
	with zipfile.ZipFile(archive, 'r') as file:
		for info in file.infolist():
			path = info.filename
			if path in want:
				item = want[path]

				name = (item[0] if not item[1] else item[1]).path
				want.pop(name)

				process_datapoint(
					item,
					file.read(info)
				)
				print(f'there are {str(len(want))} items left after this\t({name})')
	
	# check to be sure that we cound all the datapoints we wanted
	failed = 0
	for name in want:
		print(f"FAILED to find {name}")
		failed += 1
	if 0 != failed:
		raise Exception(
			f'failed to find {failed} images'
		)



def process_datapoint(datapoint, data):

	# compute some coordinates or whatever
	group = 'train' if (None == datapoint[1]) else 'val'
	datapoint = datapoint[0] if datapoint[0] else datapoint[1]
	fKey = datapoint.fKey

	is_jpg = datapoint.path.endswith('.jpg')
	jpg = f'target/yolo-dataset_{config.LIMIT}/images/{group}/{fKey}.jpg'
	txt = f'target/yolo-dataset_{config.LIMIT}/labels/{group}/{fKey}.txt'
	png = f'target/yolo-dataset_{config.LIMIT}/images/{group}/{fKey}.png'

	# delete wrong image file (if present)
	import os
	if is_jpg:
		if os.path.isfile(png):
			print(f"{fKey} had a png - oops;" + datapoint.path)
			os.remove(png)
	elif os.path.isfile(jpg):
		print(f"{fKey} had a jpg - oops;" + datapoint.path + ",  " + str(is_jpg))
		os.remove(jpg)

	# skip of it's present
	import os
	if os.path.isfile(jpg if is_jpg else png) and os.path.isfile(txt):
		return

	ensure_directory_exists(jpg)
	ensure_directory_exists(png)
	ensure_directory_exists(txt)

	import cv2
	import numpy as np

	# get the image dimenions - IIRC this was faster than PIL
	# ... note the h,w ordering ... not my idea
	image = cv2.imdecode(
		np.frombuffer(data, dtype=np.uint8),
		cv2.IMREAD_COLOR)
	ih, iw, _ = image.shape
	
	dw = 1.0 / float(iw)
	dh = 1.0 / float(ih)
	
	# copy the image to disk - this should deal with the iCCN profile issues and corrup jpegs ... maybe ...
	cv2.imwrite(jpg if is_jpg else png, image)

	# convert/write the labels - i'm assuming that they're thte same format (but we'll see)
	with open(txt, 'w') as file:
		labels = []
		for face in datapoint.patches:
			l = face.l * dw
			t = face.t * dh
			r = face.r * dw
			b = face.b * dh
			label = (f'0 {l} {t} {r} {b}\n')
			labels.append(label)
			file.write(label + '\n')

		# preview the image it we're doing a testing dataset
		if config.PREVIEW:

			for label in labels:
				l, t, r, b = list(map(float, label.split(' ')[1:]))
				
				start_point = (int(l * iw), int(t * ih))  # Top-left corner
				end_point = (int(r * iw), int(b * ih))  # Bottom-right corner
				color = (0, 255, 0)  # Green color
				thickness = 2  # Thickness of 2 pixels

				cv2.rectangle(image, start_point, end_point, color, thickness)

			cv2.imshow(f'{fKey} / {group}', image)
			cv2.waitKey(0)
			cv2.destroyAllWindows()



def greenlist(datapoints, archive):
	import os

	# load the greeinlisted
	listed = {}
	assert os.path.isfile('greenlist.txt')
	with open('greenlist.txt', 'r') as f:
		for line in f.readlines():
			line = line.strip()
			if '' ==line:
				continue
			key, value = line.strip().split('=')
			assert ('t' == value) or ('f' == value)
			listed[key] = 't' == value

	def loop():
		for item in datapoints:
			if item.fKey not in listed:
				if not archive:
					if config.GREENLIST_DEFAULT_INCLUE:
						yield item
					continue
				else:
					listed[item.fKey] = prompt_the_human(item)
					with open('greenlist.txt', 'a') as f:
						f.write(item.fKey + '=' + ('t\n' if listed[item.fKey] else 'f\n'))

			if listed[item.fKey]:
				yield item
	
	def prompt_the_human(item):
		import cv2
		import zipfile
		import numpy as np

		with zipfile.ZipFile(archive, 'r') as zip_file:
			# Read the image file from the ZIP
			with zip_file.open(item.path) as image_file:
				# Convert the image file into a numpy array
				image_data = np.frombuffer(image_file.read(), np.uint8)
				# Decode the image data into an OpenCV image
				image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)


		# Check if the image was loaded successfully
		if image is None:
			raise Exception("Error: Could not load image.")
		else:

			for face in item.patches:
				# Draw a rectangle on the image
				start_point = (face.l, face.t)  # Top-left corner
				end_point = (face.r, face.b)  # Bottom-right corner
				color = (0, 255, 0)  # Green color in BGR
				thickness = 3  # Thickness of the rectangle border

				cv2.rectangle(image, start_point, end_point, color, thickness)


			# Display the image in a window
			cv2.imshow(
				'ESC - Abort, SPACE = Reject, ENTER = Accept',
				image
			)

			accept = False
			# Wait for a key press
			key = cv2.waitKey(0)

			# Check which key was pressed
			if key == 32:  # Space key
				accept = False
			elif key == 13:  # Enter key
				accept = True
			elif key == 27:  # Escape key
				exit()

			# Destroy all OpenCV windows
			cv2.destroyAllWindows()
			return accept
	 
	
	for item in loop():
		yield item