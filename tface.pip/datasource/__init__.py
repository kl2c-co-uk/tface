
import os
import hashlib
import zipfile
import requests
import os
import traceback



class val:
	"""it's like a "property" but it is only ever run once
	"""

	def __init__(self, func):
		self.func = func
		self.func_name = func.__name__

	def __get__(self, obj, cls):
		if obj is None:
			return self

		# Call the function to get the value
		value = self.func(obj)

		# Replace the method with the value in the instance's dictionary
		setattr(obj, self.func_name, value)
		return value

def md5(input_string):
	# Encode the string to bytes, then create an MD5 hash object
	md5_hash = hashlib.md5(input_string.encode())

	# Get the hexadecimal representation of the hash
	return md5_hash.hexdigest()


todo_seen = {}

def todo(message):
	message = str(message)
	try:
		raise Exception(message)
	except Exception as e:
		stack = traceback.format_stack()[:-1]
		message = '\nTODO: '+message+'\n'

		for frame in stack:
			message += str(frame)
		key = md5(message)

		if not key in todo_seen:
			todo_seen[key] = True
			print(message)




class literator():
	def __init__(self, list):
		self._next = 0
		self._list = list
	
	def more(self):
		return self._next < len(self._list)
	
	def take(self):
		item = self._list[self._next]
		self._next += 1
		return item


def ensure_directory_exists(file_path):
	directory = os.path.dirname(file_path)
	if not os.path.exists(directory):
		os.makedirs(directory)
	return directory
		
class Blurb:
	def __init__(self, **kwargs):
		for key, value in kwargs.items():
			setattr(self, key, value)


class Cache():
	def __init__(self, target):
		self.target = target


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
				raise Exception('too many entries match `{name}`')
			else:
				found = True
				yield data
		if not found:
			raise Exception(f'no entry matched `{name}`\n\t... in `{self.zip_path}`\n')

	def text(self, name):
		for data in self.read(name):
			# turn it into a more conventional iterator
			yield literator(data.decode('utf-8').splitlines())
