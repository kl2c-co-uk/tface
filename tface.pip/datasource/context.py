
from .base import *
import requests

import os

class Cache():
	def __init__(self, target, size, heat_scale):
		self.target = target
		self.size = size
		self.heat_scale = heat_scale


	def download(self, url, name = None):
		if None == name:
			name = md5(url)
		
		save_path = self.target +name
		
		ensure_directory_exists(save_path)

		if os.path.exists(save_path):
			# print(f"The file '{save_path}' already exists. Skipping download.")
			return save_path

		print('downloading to ' + name +' this could be loonnngggg .....')
		
		response = requests.get(url, stream=True)
		with open(save_path, 'wb') as f:
				for chunk in response.iter_content(chunk_size=8192):
						if chunk:
								f.write(chunk)
		
		print(f"Downloaded {url} to {save_path}")
		return save_path

class ZipWalk():
	def __init__(self, zip_path):
		self.zip_path = zip_path
		assert os.path.isfile(zip_path)

	def find(self, test):
		with zipfile.ZipFile(self.zip_path, 'r') as file:
			for info in file.infolist():
				name = info.filename
				if test(name):
					yield (name, file.read(info))


	def read(self, name):
		if not name.startswith('/'):
			name = '/' + name

		found = False
		for _, data in self.find(lambda a : a.endswith(name)):
			if found:
				throw('too many entries match `{name}`')
			else:
				found = True
				yield data
		if not found:
			throw(f'no entry matched `{name}`\n\t... in `{self.zip_path}`\n')

	def text(self, name):
		for data in self.read(name):
			# turn it into a more conventional iterator
			yield literator(data.decode('utf-8').splitlines())