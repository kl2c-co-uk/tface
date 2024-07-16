

from datasource.base import Cache, md5

import datasource.config as config

PERTURB_ONLY_FACE = False # copy the faces and zoom into them


# PERTURB_WIPE_FACE = True # copy the image and cover teh faces with randomness

def yset_training(cache):
	for e in yset(cache, [wider.trainingFrames(cache)]):
		yield e

def yset_validate(cache):
	for e in yset(cache, [wider.validateFrames(cache)]):
		yield e

def yset(cache, framess):
	points = 0
	import datasource.config 

	import datasource.wider as wider
	import datasource.data as data
	for frames in framess:
		for frame in frames:

			# yield the full point
			yield data.DataPoint(cache, frame)
			points += 1
			if datasource.config.LIMIT > 0 and points >= datasource.config.LIMIT:
				return

			# do any perturbations of the point
			faces = frame._faces
			image = frame._jpeg.data
			if PERTURB_ONLY_FACE:
				for face in faces:
					import cv2

					x, y = face.x, face.y
					w, h = face.w, face.h

					s = config.target_height/float(h)

					i = int(s * w)
					j = int(s * h)

					if i < 100 or j < 100:
						continue

					only_face = image[y:y+h, x:x+w]

					scalred_face = cv2.resize(only_face, (i, j))


					pert_jpegi = data.JPEGImage(lambda : scalred_face)

					pert_faces = [
						data.FacePatch(
							pert_jpegi,
							x=0, y=0, w=i, h=j
						)
					]

					pert_frame =  data.FaceFrame(
						md5(frame._bound + str(points)),
						pert_jpegi,
						pert_faces
					)

					# yield the pert point
					yield data.DataPoint(cache, pert_frame)
					points += 1
					if datasource.config.LIMIT > 0 and points >= datasource.config.LIMIT:
						return




if __name__ == '__main__':
	"""
	silly block of code to "step on" the datasets
	"""

	cache = Cache('target/')

	import datasource.wider as wider

	for y in yset_training(cache):
		print(f'---\n{y.cache}')
	for y in yset_validate(cache):
		print(f'---\n{y.cache}')
	
	print('whole new dataset loaded')

	raise 'the heat maps are being overscaled'
