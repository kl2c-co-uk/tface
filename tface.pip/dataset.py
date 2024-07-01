"""

 λ nodemon --ignore target/ dataset.py

 """

target_width = 1920
target_height = 1080
heatmap_scale = 0.2

import os
import requests
import zipfile
import cv2
from PIL import Image
from io import BytesIO
import hashlib
import math

from u import *


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

	# do the training dataset
	# wider_dataset(
	import dataset_wider
	dataset_wider.wider_dataset(
		target = 'target/masked-dataset/',
		labels = 'wider_face_train_bbx_gt.txt',
		archive = 'target/wider.training.zip',
		group = 'train',
	)

	throw('??? - to it all for the other image set!')

def wider_dataset(
	target = 'target/masked-dataset/',
	labels = 'wider_face_train_bbx_gt.txt',
	archive = 'target/wider.training.zip',
	group = 'train'
):
	def wider_datum(args):
		image, faces, bound = args
		# compute teh jpg name we'll use
		jpg = f'{target}{group}/images/{bound}.jpg'
		for _, data in zipfile_get(archive, image):
			# repack image (and the faces)
			(faces, scaled) = repack_image(faces, data, target_width, target_height)

			# skip images/masks that already exist
			if not os.path.isfile(jpg):
				# save the repacked image
				ensure_directory_exists(jpg)
				scaled.save(jpg)

		png = f'{target}{group}/heatmap/{bound}.png'
		if not os.path.isfile(png):
			heatmap = Image.new('L', (
				int(target_width * heatmap_scale),
				int(target_height * heatmap_scale)))

			# compute the bounds for the hot-spots
			def hot_spot(face):
				fx, fy, fw, fh = face

				assert fw > 0, "bad width in {image}"
				assert fh > 0

				half_w = float(fw) / 2.0
				half_h = float(fh) / 2.0

				return Bunch(
					off_x = float(fx) + half_w,
					off_y = float(fy) + half_h,
					scale_x = 1.0 / half_w,
					scale_y = 1.0 / half_h,
					edge_l = fx,
					edge_r = fx + fw,
					edge_b = fy,
					edge_t = fy + fh,
				)

			hot_spots = list(map(hot_spot, faces))

			# find the max heat per pixel
			for x in range(0, heatmap.size[0]):
				x = x / heatmap_scale
				for y in range(0, heatmap.size[1]):
					y = y / heatmap_scale
					heat = 0.0

					for spot in hot_spots:
						if spot.edge_l <= x and x <= spot.edge_r:
							if spot.edge_b <= y and y <= spot.edge_t:
								pix_x = (x - spot.off_x) * spot.scale_x
								pix_y = (y - spot.off_y) * spot.scale_y

								piz_sq = (pix_x * pix_x) + (pix_y * pix_y)

								if piz_sq <= 1:
									heat = max(heat, 1.0 - math.sqrt(piz_sq))
					heatmap.putpixel((
						int(x*heatmap_scale),int(y*heatmap_scale)), int(256.0 * heat))

			#save the heat-map
			ensure_directory_exists(png)
			heatmap.save(png)

	# batch = []
	for image, faces in wider_faces(labels):
		bound = image
	# 	batch.append((image, faces, bound))
		import dataset_wider
		dataset_wider.process_single_item((target, labels, archive, group, image, faces, bound))

	# for datum in batch:
	# 	wider_datum( datum )
	

	
def wider_faces(labels):
	# we start with the annotations file (sorry)
	for _, text in zipfile_get('target/wider.annotations.zip', labels):
		
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

					x, y, w, h = tuple(map(int, (x, y, w, h)))

					if h <= 0 or w <= 0:
						print(f'found a zero-face in the data for `{image}` and i am skipping it')
						count -= 1
					elif int(w * heatmap_scale) <= 0 or int(h * heatmap_scale) <= 0:
						print(f'i will smoosh one of the faces in `{image}` so i am skipping it')
						count -= 1
					else:

						assert w > 0
						assert h > 0
						assert int(w * heatmap_scale) > 0
						assert int(h * heatmap_scale) > 0

						faces.append((x, y, w, h))
			
			# we don't need to mess with them here
			yield (image, faces)

def repack_image(faces, data, target_width, target_height):
	import random
	target_image = Image.new('RGB', (target_width, target_height), tuple(random.randint(0, 255) for _ in range(3)))

	image = Image.open(BytesIO(data))

	(width, height) = image.size

	# find the uniform scale factor and compute the w/h
	scale_factor = min(
		float(target_width) / width,
		float(target_height) / height)
	scaled_w = int(scale_factor * width)
	scaled_h = int(scale_factor * height)


	# now set the x/y used to place the image in the final image
	scaled_offset_x = 0
	scaled_offset_y = 0
	if scaled_w < target_width:
		scaled_offset_x = random.randint(0, target_width - scaled_w)
	if scaled_h < target_height:
		scaled_offset_y = random.randint(0, target_height - scaled_h)

	assert scaled_offset_x >= 0
	assert scaled_offset_y >= 0

	# scale the image and overlay image
	target_image.paste(
		image.resize((scaled_w, scaled_h), Image.LANCZOS),
		(scaled_offset_x, scaled_offset_y))

	# now scale faces
	def scale_face(old_face):
		(old_x, old_y, old_w, old_h) = old_face

		newx = (old_x * scale_factor) + scaled_offset_x
		newy = (old_y * scale_factor) + scaled_offset_y
		neww = (old_w * scale_factor)
		newh = (old_h * scale_factor)

		return (newx, newy, neww, newh)
	aspect = float(scaled_w) / width
	scaled_faces = list(map(scale_face, faces))

	# return what we've got!
	return (scaled_faces, target_image) 

if __name__ == "__main__":
	build_dataset()
	throw("data set loaded - okie dokee")


