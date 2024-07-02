

import multiprocessing

SMALL_LIMIT = -1
COUNT_FORKS = multiprocessing.cpu_count() * 4
FORKIT = True
RANDOM_FILLER = True

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

class FacePatch:
	def __init__(self, face):
		self.face = face

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

def forked(args):
	
	cache, out, images, datapoint = args
	
	bound = datapoint.jpeg_path
	jpg = f'{out}/images/{bound}.jpg'
	png = f'{out}/heatmap/{bound}.png'
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

	# see how far we'll wiggle it - offset the faces
	import random
	bump = (random.randint(0, cache.size[0] - the_image.size[0]), random.randint(0, cache.size[1] - the_image.size[1]))

	# Create a new RGB image
	blank = Image.new('RGB', cache.size, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

	# fill it with random colours
	if RANDOM_FILLER:
		for x in range(blank.size[0]):
			for y in range(blank.size[1]):
				blank.putpixel((x, y), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
	
	# make the changes
	faces = reFace(faces, bump)
	blank.paste(the_image, bump)
	the_image = None
	ensure_directory_exists(jpg)
	blank.save(jpg)

	###
	# make the heat-map

	
	patches = list(map(lambda face: FacePatch(face), faces))

	# create a new black and white heatmap image
	blank = Image.new('L', (int(cache.size[0] * cache.heat_scale), int(cache.size[1] * cache.heat_scale)), 0)
	for x in range(blank.size[0]):
		for y in range(blank.size[1]):
			heat = 0
			for patch in patches:
				heat = max(heat, patch.heat(x / cache.heat_scale, y / cache.heat_scale))
			heat *= 256.0
			heat = int(heat)
			blank.putpixel((x, y), heat)
	ensure_directory_exists(png)
	blank.save(png)

	print( datapoint.jpeg_path )

def main(cache, out):

	# download the annotations file
	annotations = cache.download(
		'http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip'
	)

	def extract(txt,url,out):
		datapoints = []
		for lines in ZipWalk(annotations).text(txt):
			while lines.more() and (((SMALL_LIMIT) > len(datapoints)) or (-1 == SMALL_LIMIT)):
				datapoints.append(chomp__datapoint(lines))

		# download the images
		images = cache.download(url)
		widen = lambda datapoint: (cache, out, images, datapoint)
		todo = list(map(widen, datapoints))
		if FORKIT:
			with multiprocessing.Pool(processes=COUNT_FORKS) as pool:
				pool.map(forked, todo)
		else:
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
