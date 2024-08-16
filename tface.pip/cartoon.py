
from datasource import Blurb, Cache, md5, ZipWalk, ensure_directory_exists, FacePatch, DataPoint

import datasource.config as config



def main():
	cache = Cache('target/')
	i_cartoon(cache)



def split_export(datapoints, l, r, archive):
	raise Exception(
		'???'
	)

def i_cartoon(cache):

	# all the datapoints
	datapoints = i_cartoon_datapoints(cache)

	# split them 80:10
	split = random_split(datapoints, 8, 1)
	
	# limit ourselves (sorry)
	todo = only(split, config.LIMIT)

	# extract the datasets
	


	for datapoint in todo:


		# compute some coordinates or whatever
		group = 'train' if (None == datapoint[1]) else 'val'
		datapoint = datapoint[0] if datapoint[0] else datapoint[1]
		fKey = md5(datapoint.path)
		jpg = f'target/yolo-dataset_{config.LIMIT}/images/{group}/{fKey}.jpg'
		txt = f'target/yolo-dataset_{config.LIMIT}/labels/{group}/{fKey}.txt'

		# TODO; skip of it's present

		# grab the archive
		training = cache.download(
			# training = target/cb67961c4ba344c84b3e5442206436ac
			'https://drive.usercontent.google.com/download?id=1xXpE0qs2lONWKL5dqaFxqlJ_t5-glNpg&export=download&authuser=0&confirm=t&uuid=f6f6beb7-4c3b-40a7-b52d-12c62c2e84fe&at=APZUnTV9QwxtWfOsgjgqW-7icoaM:1723671279280'
		)

		ensure_directory_exists(jpg)
		ensure_directory_exists(txt)

		for data in ZipWalk(training).read(datapoint.path):
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
			
			# copy the image to disk
			with open(jpg, 'wb') as file:
				file.write(data)

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

					cv2.imshow('image with boxes', image)
					cv2.waitKey(0)
					cv2.destroyAllWindows()

def i_cartoon_datapoints(cache):
	annotations = cache.download(
		# annotations = target/712e3f96290bfc9c1c93a18f16ef40e8
		'https://drive.usercontent.google.com/download?id=15IHSlNBZBZs_hj6B341swc00ha5fpvB7&export=download&authuser=0&confirm=t&uuid=72fd55fe-6a76-4c73-91ee-63de54aa2775&at=APZUnTXR3ogM4tIFCGrcFoaswAor:1723670943322'
	)


	for lines in ZipWalk(annotations).text('personai_icartoonface_dettrain_anno_updatedv1.0.csv'):
		seen = []

		# start a non-datapoint
		last = ''
		data = '?unset?'

		while lines.more():
			line = lines.take()

			# get the line content
			name, l, t, r, b = line.split(',')

			# check if we need to switch datapoints
			if name != last:

				# emit the prior datapoint
				if '' != last:					
					yield DataPoint(path = last, patches = data)					
				
				# start a datapoint
				data = []
				last = name

				# check the names. maybe
				if name in seen:
					raise Exception(
						'the items are not grouped as i d expected'
					)
				seen.append(name)
			
			# add the patch tot he datapoint
			data.append(
				FacePatch(ltrb = [l, t, r, b])
			)


		
		# yield the final datapooint
		if '' != last:					
			yield DataPoint(path = last, patches = data)							


def random_split(data, l, r, seed = 41):
	t = l + r

	import random
	random = random.Random(seed)

	for item in data:
		if random.randint(0, t) < l:
			yield (item, None)
		else:
			yield (None, item)


def only(list, count):
	if count <= 0:
		for item in list:
			yield item
	else:
		for item in list:
			if 0 < count:
				yield item
				count -= 1



if '__main__' == __name__:
	main()
