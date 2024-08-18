
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


from datasource import Blurb, Cache, md5, ZipWalk, ensure_directory_exists, random_split, only
import datasource.config as config
from datasource import random_split, only
from PIL import Image


def split_export(datapoints, train, val, archive):

	# split them train:val
	split = random_split(datapoints, train, val)
	
	# limit ourselves (for testing)
	todo = only(split, config.LIMIT)

	for datapoint in todo:

		# compute some coordinates or whatever
		group = 'train' if (None == datapoint[1]) else 'val'
		datapoint = datapoint[0] if datapoint[0] else datapoint[1]
		fKey = md5(datapoint.path)

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
			continue


		ensure_directory_exists(jpg)
		ensure_directory_exists(png)
		ensure_directory_exists(txt)

		for data in ZipWalk(archive).read(datapoint.path):
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

