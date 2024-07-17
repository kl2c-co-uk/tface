

import multiprocessing
import numpy


import datasource.data as data
import datasource.config as config
from datasource import *

def trainingFrames(cache):
	for frame in frames(cache,
		'wider_face_train_bbx_gt.txt',
		'https://drive.usercontent.google.com/download?id=15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M&export=download&authuser=0&confirm=t&uuid=6d1b1482-0707-4fee-aca1-0ea41ba1ecb6&at=APZUnTX8U1BtsQRxJTqGH5qAbkFf%3A1719226478335',
	):
		yield frame


def validateFrames(cache):
	for frame in frames(cache,
		'wider_face_val_bbx_gt.txt',
		'https://drive.usercontent.google.com/download?id=1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q&export=download&authuser=0&confirm=t&uuid=8afa3062-ddbc-44e5-83fd-c4e1e2965513&at=APZUnTUX4c1Le0kpmfMNJ6i3cIJh%3A1719227725353',
	):
		yield frame

def frames(cache, txt, url):
	# download the annotations file
	annotations = cache.download(
		'http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip'
	)

	# download the images file
	images = cache.download(url)
	
	for lines in ZipWalk(annotations).text(txt):
		while lines.more():
			jpeg_path = lines.take()
			
			jpegImage = data.JPEGImage(lambda : frame_jpeg(images, jpeg_path))

			face_count = int(lines.take())

			# read entries
			faces = []
			if 0 == face_count:
				# for extra weirdness; entries (or The One Entry) with no faces have a line with garbage data
				blank = lines.take().strip()
				if '0 0 0 0 0 0 0 0 0 0 '.strip() != blank:
					raise Exception('empty entry had a funky line!')
			else:
				while len(faces) < face_count:
					# grab the first four and convert them to ints
					x, y, w, h = map(int, lines.take().split(' ')[:4])

					# skip ones that're too small (this is also done later, but, it's nice to do these early)
					if (
						w < config.MIN_WIDTH
					) or (
						h < config.MIN_HEIGHT
					) or (
						(w * h) < config.MIN_SIZE
					):
						face_count -= 1
					else:

						assert w > 0
						assert h > 0

						faces.append(
							data.FacePatch(
								jpegImage,
								x=x, y=y, w=w, h=h
							)
						)

			# just check that face count didn't get weird
			assert 0 <= face_count
			
			yield data.FaceFrame(
				md5(jpeg_path),
				jpegImage,
				faces
			)



def frame_jpeg(images, path):
	for data in ZipWalk(images).read(path):
		import cv2
		import numpy as np 
		return cv2.imdecode(
			np.frombuffer(data, dtype=np.uint8),
			cv2.IMREAD_COLOR)
		


