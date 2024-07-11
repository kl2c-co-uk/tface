

import multiprocessing
import numpy

SMALL_LIMIT = 2

from . import *
from .base import *
from .context import *

print('why are the heat-maps the wrong size?')

class FacePatch:
	def __init__(self, face):
		self.face = face

		self.bound_l = face.x
		self.bound_r = face.x + face.w

		self.bound_b = face.y
		self.bound_t = face.y + face.h

		self.half_w = float(face.w) / 2.0
		self.half_h = float(face.h) / 2.0

		self.center_x = face.x + self.half_w
		self.center_y = face.y + self.half_h

		if 0.0 == self.half_w:
			self.scale_x = 1.0 / 0.00004 # IDK; this should be so small it doesn't count
		else:
			self.scale_x = 1.0 / self.half_w

		if 0.0 == self.half_h:
			self.scale_y = 1.0 / 0.00004 # IDK; this should be so small it doesn't count
		else:
			self.scale_y = 1.0 / self.half_h

	def over(self, x, y):
		return (self.bound_l <= x) and (x <= self.bound_r) and (self.bound_b <= y) and (y <= self.bound_t)

	def heat(self, x, y):

		x = (x - self.center_x) * self.scale_x
		x *= x

		y = (y - self.center_y) * self.scale_y
		y *= y

		r = x + y

		if r >= 1:
			return 0
		else:
			import math
			return 1.0 - math.sqrt(r)

def chomp__datapoint(lines):
	jpeg_path = lines.take()
	face_count = int(lines.take())

	# read entries
	faces = []
	if 0 == face_count:
		# for extra weirdness; entries (or The One Entry) with no faces have a line with garbage data
		blank = lines.take().strip()
		if '0 0 0 0 0 0 0 0 0 0 '.strip() != blank:
			throw('empty entry had a funky line!')
	else:
		while len(faces) < face_count:
			x, y, w, h, *_ = lines.take().split(' ')

			x, y, w, h = tuple(map(int, (x, y, w, h)))

			if h <= 0 or w <= 0:
				# print(f'found a zero-face in the data for `{image}` and i am skipping it')
				face_count -= 1
			else:

				assert w > 0
				assert h > 0

				faces.append(Bunch(x=x, y=y, w=w, h=h))
	assert 0<= face_count
	
	return Bunch(jpeg_path=jpeg_path, faces=faces)

def forked(args):
	
	cache, out, images, datapoint = args
	
	# turnt he name into an md5 to prevent lots of class anme silliness
	bound = md5(datapoint.jpeg_path)

	jpg = f'{out}/images/{bound}.jpg'
	png = f'{out}/masks/{bound}.png'
	faces = datapoint.faces


	# skip ones we've already done
	if os.path.isfile(jpg) and os.path.isfile(png):
		return

	###
	# load the original image

	the_image = None
	for data in ZipWalk(images).read(datapoint.jpeg_path):
		from PIL import Image
		from io import BytesIO
		assert None == the_image
		the_image = Image.open(BytesIO(data))
		assert None != the_image
	
	assert None != the_image

	###
	# resize the the_image image and faces

	# shrink width
	if the_image.size[0] > cache.size[0]:
		factor = float(cache.size[0]) / float(the_image.size[0])

		nw = int(the_image.size[0] * factor)
		nh = int(the_image.size[1] * factor)

		the_image = the_image.resize((nw, nh))

		faces = reFace(faces, factor)
	
	# shrink height
	if the_image.size[1] > cache.size[1]:
		factor = float(cache.size[1]) / float(the_image.size[1])

		nw = int(the_image.size[0] * factor)
		nh = int(the_image.size[1] * factor)

		the_image = the_image.resize((nw, nh))

		faces = reFace(faces, factor)

	###
	# composite the image

	# see how far we'll wiggle it - and offset the faces
	import random
	bump = (random.randint(0, cache.size[0] - the_image.size[0]), random.randint(0, cache.size[1] - the_image.size[1]))
	faces = reFace(faces, bump)

	# create the random image with numpy (so much faster - relevant when we have to do this over 12k times)
	background_image = Image.fromarray(
		numpy.random.randint(0, 256, (cache.size[1], cache.size[0], 3), dtype=numpy.uint8)
	)
	
	# make the changes
	background_image.paste(the_image, bump)
	the_image = background_image
	ensure_directory_exists(jpg)
	the_image.save(jpg)


	###
	# make a heat map with numpy

	# create the heat_map as all zeroes
	
	heat_w = int(cache.size[0] * cache.heat_scale)
	heat_h = int(cache.size[1] * cache.heat_scale)
	heat_map = numpy.zeros(
		# w/h are flipped because math is hard
		(heat_h, heat_w),
		dtype=numpy.uint8)
	
	# re-scale the faces (once more) and fill them in
	patches = list(map(lambda face: FacePatch(face), reFace(faces, cache.heat_scale)))

	# fill it in
	for patch in [FacePatch(face) for face in reFace(faces, cache.heat_scale)]:
		for x in range(patch.bound_l, patch.bound_r):
			for y in range(patch.bound_b, patch.bound_t):
				heat = int(patch.heat(x,y) * 256.0)
				heat_map[y, x] = max(heat_map[y, x], heat)

	ensure_directory_exists(png)
	Image.fromarray(heat_map).save(png)

	print( datapoint.jpeg_path )

def main(cache, out):

	# download the annotations file
	annotations = cache.download(
		'http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip'
	)

	def extract(txt,url,out):
		datapoints = []
		for lines in ZipWalk(annotations).text(txt):
			def is_small():
				if SMALL_LIMIT > 0:
					return len(datapoints) < SMALL_LIMIT
				else:
					return True

			while lines.more() and is_small():
				datapoints.append(chomp__datapoint(lines))

		# download the images
		images = cache.download(url)
		widen = lambda datapoint: (cache, out, images, datapoint)
		todo = list(map(widen, datapoints))
	
		for item in todo:
			forked(item)

	# scan that for training data
	extract(
		'wider_face_train_bbx_gt.txt',
		'https://drive.usercontent.google.com/download?id=15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M&export=download&authuser=0&confirm=t&uuid=6d1b1482-0707-4fee-aca1-0ea41ba1ecb6&at=APZUnTX8U1BtsQRxJTqGH5qAbkFf%3A1719226478335',
		out+'train/'
	)

	# extract validation dataset
	extract(
		'wider_face_val_bbx_gt.txt',
		'https://drive.usercontent.google.com/download?id=1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q&export=download&authuser=0&confirm=t&uuid=8afa3062-ddbc-44e5-83fd-c4e1e2965513&at=APZUnTUX4c1Le0kpmfMNJ6i3cIJh%3A1719227725353',
		out+'validation/'
	)

def reFace(faces, factor):
	"""adjusts the faces by either an offset or a scale
	"""

	offsets = (0, 0)
	scale = 1.0

	if type(offsets) == type(factor):
		x, y = factor
		offsets = (x, y)
	elif type(scale) == type(factor):
		scale = factor
	else:
		throw(' whats? ' + type(factor))

	offx, offy = offsets

	result = []
	for face in faces:
		result.append(
			Bunch(
				w = int(face.w * scale),
				h = int(face.h * scale),
				x = int(face.x* scale) + offx,
				y = int(face.y * scale) + offy,
			)
		)

	return result
