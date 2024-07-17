

from datasource import md5, Cache

import datasource.config as config

def yset_training(cache):
	import datasource.wider as wider
	for e in yset(cache, [wider.trainingFrames(cache)]):
		yield e

def yset_validate(cache):
	import datasource.wider as wider
	for e in yset(cache, [wider.validateFrames(cache)]):
		yield e

def yset(cache, framess):
	points = 0
	import datasource.config 
	import datasource.data as data
	for frames in framess:
		for frame in frames:

			# yield the full point
			full_point = data.DataPoint(cache, frame)
			yield full_point
			points += 1
			if datasource.config.LIMIT > 0 and points >= datasource.config.LIMIT:
				return

			# permutations?
			faces = full_point.faces
			image = full_point._frame._jpeg.data
			if config.PERMUTE_FACES:
				for face in faces:

					if face.is_small:
						continue

					import cv2

					x, y = face.x, face.y
					w, h = face.w, face.h

					s = config.INPUT_HEIGHT / float(h)

					i = int(s * w)
					j = int(s * h)

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


	for y in yset_training(cache):
		print(f'---\n{y.cache}')

	for y in yset_validate(cache):
		print(f'---\n{y.cache}')
	
	print('whole new dataset loaded')
