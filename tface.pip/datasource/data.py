"""

https://mermaid.live/edit#pako:eNp1kcFugzAMhl_Fypm-AIedWLXuVI1Ku-TiJQYikQQFp1NX-u5LYAIJdTf_9uffv-S7UF6TKEXT-2_VYWC4VNJJHuNXG3Do4PNUvX7AOJAyjVF5JPl9oPZksaVFHlHRGVl1i3zD8a-qzU9GVugY0NIGweHwArt-XnnWX0_m4eQdFXA1CAi9R01hWnHYZZp5dLf_-IyT00vMChnP3jhe5GaaTZwPFnszkgb20w5d5YzadBA4oHHGtTNaEz_BYsp3TZ4amTZMFMJSumV0esxdOgApuKOUVZSp1NRg7FkK6R4Jxci-vjklSg6RChGH7FYZTP-zS_PxC2sPsMo

"""


class JPEGImage:
	def __init__(self, load):
		self.load = load
		self._data = None
	
	@property
	def data(self):
		return self.load()

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
	
	@property
	def cache(self):
		import json, os, random, cv2
		import numpy as np
		import datasource.config as config

		jpg = f'{self._cache.target}cache/{self._frame._bound}.jpg'
		png = f'{self._cache.target}cache/{self._frame._bound}.png'

		# if the input and label exist ...
		if os.path.isfile(jpg) and  os.path.isfile(png):
			# return the image!
			return (jpg, png)

		# load the image
		image = self._frame._jpeg.data
		faces = self._frame._faces
		
		# numpy is height, width, channels

		# if it's too wide
		if image.shape[1] > config.target_width:
			raise 'shrink the image width'
			raise 'update the faces'
		
		# if it's too tall
		if image.shape[0] > config.target_height:

			t_width = float(image.shape[1]) / float(image.shape[0])
			t_width *= float(config.target_height)

			# resize the image
			image = cv2.resize(image,
				(int(t_width), config.target_height), interpolation=cv2.INTER_AREA)

			# scale the faces
			faces = map(lambda face: face.scale(t_width / float(image.shape[1])), faces)

		# compute an offset
		o_x = random.randint(0, config.target_width - image.shape[1])
		o_y = random.randint(0, config.target_height - image.shape[0])
		faces = map(lambda face: face.bump(o_x, o_y), faces)

		# splat the image
		
		# create the random image with numpy (so much faster - relevant when we have to do this over 12k times)
		under = np.random.randint(
			0, 256,
			(config.target_height, config.target_width, 3),
			dtype=np.uint8)

		# paste it over
		oh, ow = image.shape[:2]
		under[o_y:o_y+oh, o_x:o_x+ow] = image
		image = under
		under = None

		# create the bad heat .png
		bheat = np.zeros(
			(int(config.target_height * config.heatmap_scale), int(config.target_width * config.heatmap_scale)),
			dtype=np.uint8)
		for face in map(lambda f: f.scale(config.heatmap_scale), faces):
			x, y = face.x, face.y 
			w, h = face.w, face.h

			bheat[y:y+h, x:x+w] = 255

		# store the things
		from datasource.base import ensure_directory_exists
		ensure_directory_exists(jpg)
		ensure_directory_exists(png)
		cv2.imwrite(jpg, image)
		cv2.imwrite(png, bheat)

		# recur
		return self.cache


