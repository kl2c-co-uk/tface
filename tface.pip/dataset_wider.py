"""
the WIDER face dataset

"""

from u import *

from dataset import *

def wider_dataset(
	target,
	labels,
	archive,
	group
):
	batch = []
	for image, faces in wider_faces(labels):
		bound = image
		batch.append((target, labels, archive, group, image, faces, bound))

	throw('ohh kay')

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
						# print(f'found a zero-face in the data for `{image}` and i am skipping it')
						count -= 1
					elif int(w * heatmap_scale) <= 0 or int(h * heatmap_scale) <= 0:
						# print(f'i will smoosh one of the faces in `{image}` so i am skipping it')
						count -= 1
					else:

						assert w > 0
						assert h > 0
						assert int(w * heatmap_scale) > 0
						assert int(h * heatmap_scale) > 0

						faces.append((x, y, w, h))
			
			# we don't need to mess with them here
			yield (image, faces)

def process_single_item(args):
	target, labels, archive, group, image, faces, bound = args
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







