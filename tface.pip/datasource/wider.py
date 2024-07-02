
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
	
	# TODO; skip old ones
	print(datapoint.jpeg_path)
	print(datapoint.jpeg_path)
	print(datapoint.jpeg_path)

	raise('locate the original image')
	raise('resize the original image')
	raise('save the resized image to disk')

	raise('build a heat-map')
	
	
	raise('>??>? '+str(args))
