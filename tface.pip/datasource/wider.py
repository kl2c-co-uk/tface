
from . import *
from .base import *

from .context import *

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



SMALL_LIMIT = 3


def foo(cache, out):

	# download the annotations file
	annotations = cache.download(
		'http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip'
	)

	# scan that for training data
	datapoints = []
	for lines in ZipWalk(annotations).text('wider_face_train_bbx_gt.txt'):
		while lines.more() and (SMALL_LIMIT*2) > len(datapoints):
			datapoints.append(chomp__datapoint(lines))

	# download the images
	images = cache.download(
		'https://drive.usercontent.google.com/download?id=15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M&export=download&authuser=0&confirm=t&uuid=6d1b1482-0707-4fee-aca1-0ea41ba1ecb6&at=APZUnTX8U1BtsQRxJTqGH5qAbkFf%3A1719226478335',
	)

	# do the first one ... just to be sage
	forked(
		list(map(lambda datapoint: (cache, out+'train/', images, datapoint), datapoints))[0]
	)

	throw('i ahve done onw')

	import multiprocessing
	with multiprocessing.Pool(processes=SMALL_LIMIT) as pool:
		pool.map(forked, list(map(lambda datapoint: (cache, out+'train/', images, datapoint), datapoints)))

	throw('done that fork thing!!!')

def forked(args):
	
	cache, out, images, datapoint = args
	
	bound = datapoint.jpeg_path
	jpg = f'{out}/images/{bound}.jpg'
	png = f'{out}/heatmap/{bound}.png'
	faces = datapoint.faces

	def reFace(faces, factor):

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

	# TODO; skip old ones
	print(datapoint.jpeg_path)
	print(datapoint.jpeg_path)
	print(datapoint.jpeg_path)

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
		throw ('??? shrink width')
	
	# shrink height
	if the_image.size[1] > cache.size[1]:
		factor = float(cache.size[1]) / float(the_image.size[1])

		nw = int(the_image.size[0] * factor)
		nh = int(the_image.size[1] * factor)

		the_image = the_image.resize((nw, nh))

		faces = reFace(faces, factor)

	###
	# composite the face

	# see how far we'll wiggle it - offset the faces
	import random
	bump = (random.randint(0, cache.size[0] - the_image.size[0]), random.randint(0, cache.size[1] - the_image.size[1]))

	# Create a new RGB image
	blank = Image.new('RGB', cache.size, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

	# fill ti with random colours
	print("ddot: random colours")
	# for x in range(blank.size[0]):
	# 	for y in range(blank.size[1]):
	# 		blank.putpixel((x, y), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
	
	# make the changes
	faces = reFace(faces, bump)
	blank.paste(the_image, bump)
	the_image = None
	ensure_directory_exists(jpg)
	blank.save(jpg)

	def enPatch(face):
		class patch:
			def __init__(self, face):
				self.face = face

				self.half_w = float(face.w) / 2.0
				self.half_h = float(face.h) / 2.0

				self.center_x = face.x + self.half_w
				self.center_y = face.y + self.half_h

				if 0.0 == self.half_w:
					throw('??? - ohnoes')
				else:
					self.scale_x = 1.0 / self.half_w
				if 0.0 == self.half_h:
					throw('??? - ohnoes')
				else:
					self.scale_y = 1.0 / self.half_h


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
		return patch(face)

	patches = list(map(enPatch, faces))

	# create a new black and white image
	blank = Image.new('L', (int(cache.size[0] * cache.heat_scale), int(cache.size[1] * cache.heat_scale)), 0)
	for x in range(blank.size[0]):
		for y in range(blank.size[1]):
			heat = 0
			for patch in patches:
				heat = max(heat, patch.heat(x / cache.heat_scale, y / cache.heat_scale))
			blank.putpixel((x, y), int(256.0 * heat))
	ensure_directory_exists(png)
	blank.save(png)
	throw ('that should be ONE')
