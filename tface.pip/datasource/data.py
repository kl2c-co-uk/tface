"""

https://mermaid.live/edit#pako:eNp1kcFugzAMhl_Fypm-AIedWLXuVI1Ku-TiJQYikQQFp1NX-u5LYAIJdTf_9uffv-S7UF6TKEXT-2_VYWC4VNJJHuNXG3Do4PNUvX7AOJAyjVF5JPl9oPZksaVFHlHRGVl1i3zD8a-qzU9GVugY0NIGweHwArt-XnnWX0_m4eQdFXA1CAi9R01hWnHYZZp5dLf_-IyT00vMChnP3jhe5GaaTZwPFnszkgb20w5d5YzadBA4oHHGtTNaEz_BYsp3TZ4amTZMFMJSumV0esxdOgApuKOUVZSp1NRg7FkK6R4Jxci-vjklSg6RChGH7FYZTP-zS_PxC2sPsMo

"""

SHRINK = True
SPLAT = True
SCALE_HEAT = True


import datasource.config as config
from datasource import val, todo


class JPEGImage:
	def __init__(self, load):
		self.load = load
	
	@val
	def data(self):
		image = self.load()
		image.flags.writeable = False
		return image

class FacePatch:
	def __init__(self, jpeg, x, y, w, h):
		assert JPEGImage == type(jpeg)
		self.jpeg = jpeg
		self.x = x
		self.y = y
		self.w = w
		self.h = h
	
	def bump(self, i, j):
		return FacePatch(
			self.jpeg,
			self.x + i, self.y + j,
			self.w, self.h
		)

	def scale(self, v):
		return FacePatch(
			self.jpeg,
			int(self.x * v), int(self.y * v),
			int(self.w * v), int(self.h * v)
		)

	@val
	def grid(self):
		return self.w * self.h
	
	@val
	def is_small(self):
		"""
		this checks to see if the face is too small

		it assumes that this image is 1920x1080 (which is bad)
		"""

		return (
			self.w < config.MIN_WIDTH
		) or (
			self.h < config.MIN_HEIGHT
		) or (
			self.grid < config.MIN_SIZE
		)



class FaceFrame:
	def __init__(self, md5id, image, faces):
		assert JPEGImage == type(image)
		self._bound = md5id
		self._jpeg = image
		self._faces = faces

class DataPoint:
	def __init__(self, cache, frame):
		self._cache = cache
		self._frame = frame
		self._faces = None

	@val
	def faces(self):
		if self._faces:
			return self._faces
		
		faces = sorted(self._frame._faces, key =lambda face: face.grid)[::-1]
		
		faces1 = []
		for face in faces:
			if face.is_small:
				continue
			else:
				faces1.append(face)

		self._faces = faces1
		return self._faces


	
	@property
	def cache(self):
		import json, os, random, cv2
		import numpy as np

		jpg = f'{self._cache.target}cache/{self._frame._bound}.jpg'
		png = f'{self._cache.target}cache/{self._frame._bound}.png'
		jsl = f'{self._cache.target}cache/{self._frame._bound}.jsl'

		# if the input and label exist ...
		if os.path.isfile(jpg) and os.path.isfile(png) and os.path.isfile(jsl):
			# return the image!
			return (jpg, png, jsl)

		# load the image
		image = self._frame._jpeg.data
		faces = self.faces
		
		###
		# shrink the image
		if SHRINK:
			# if it's too wide
			if image.shape[1] > config.INPUT_WIDTH:
				v = float(config.INPUT_WIDTH) / float(image.shape[1])

				i = int(image.shape[1] * v)
				j = int(image.shape[0] * v)
				
				# resize the image
				image = cv2.resize(image, (i,j), interpolation=cv2.INTER_AREA)

				# scale the faces
				faces = map(lambda face: face.scale(v), faces)
			
			# if it's too tall
			if image.shape[0] > config.INPUT_HEIGHT:
				v = float(config.INPUT_HEIGHT) / float(image.shape[0])

				i = int(image.shape[1] * v)
				j = int(image.shape[0] * v)
				
				# resize the image
				image = cv2.resize(image, (i,j), interpolation=cv2.INTER_AREA)

				# scale the faces
				faces = map(lambda face: face.scale(v), faces)

		# splat the image
		if SPLAT:
			assert SHRINK
			# compute an offset
			o_x = random.randint(0, config.INPUT_WIDTH - image.shape[1])
			o_y = random.randint(0, config.INPUT_HEIGHT - image.shape[0])
			faces = map(lambda face: face.bump(o_x, o_y), faces)

			# create the random image with numpy (so much faster - relevant when we have to do this over 12k times)
			under = np.random.randint(
				0, 256,
				(config.INPUT_HEIGHT, config.INPUT_WIDTH, 3),
				dtype=np.uint8)

			# paste it over
			oh, ow = image.shape[:2]
			under[o_y:o_y+oh, o_x:o_x+ow] = image
			image = under
			under = None

		# need to perminise this
		faces = list(faces)

		# create the bad heat .png
		bheat = np.zeros(
			((int(image.shape[0] * config.HEATMAP_SCALE), int(image.shape[1] * config.HEATMAP_SCALE)) if SCALE_HEAT else (image.shape[0], image.shape[1])),
			dtype=np.uint8)

		for face in map(lambda face: face.scale(config.HEATMAP_SCALE), faces) if SCALE_HEAT else faces:
			x, y = face.x, face.y 
			w, h = face.w, face.h

			bheat[y:y+h, x:x+w] = 255


		# store the image(s)
		from datasource.base import ensure_directory_exists
		ensure_directory_exists(jpg)
		ensure_directory_exists(png)
		cv2.imwrite(jpg, image)
		cv2.imwrite(png, bheat)

		# create+store the json labels
		label = []
		for face in faces:
			tw = 1.0 / float(config.INPUT_WIDTH)
			th = 1.0 / float(config.INPUT_HEIGHT)
			label.append({
				'x': (face.w * tw),
				'y': (face.w * th),
				'w': (face.w * tw),
				'h': (face.w * th),
			})	
		with open(jsl, 'w') as f:
			json.dump(label, f, indent=1)

		# recur
		return self.cache


