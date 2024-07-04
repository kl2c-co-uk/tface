print('2024-07-03; this is old and should probably be removed')
import os
import hashlib
import zipfile

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

def md5(input_string):
    # Encode the string to bytes, then create an MD5 hash object
    md5_hash = hashlib.md5(input_string.encode())

    # Get the hexadecimal representation of the hash
    return md5_hash.hexdigest()

def ensure_directory_exists(file_path):
	directory = os.path.dirname(file_path)
	if not os.path.exists(directory):
		os.makedirs(directory)
	return directory
				
class Bunch:
	def __init__(self, **kwds):
		self.__dict__.update(kwds)

def zipfile_get(file, name):
	found = False
	for item in zipfile_all(file, lambda a : a.endswith(name)):
		if found:
			throw('too many entries match `{name}`')
		else:
			found = True
			yield item
	if not found:
		throw(f'no entry matched `{name}`')

def zipfile_all(file, test):
	with zipfile.ZipFile(file, 'r') as file:
		for info in file.infolist():
			name = info.filename
			if test(name):
				yield (name, file.read(info))

def download_file(url, save_path):

	if None == save_path:
		name = 'target/' + md5(url)
		download_file(url, name)
		return name
 	
	ensure_directory_exists(save_path)

	if os.path.exists(save_path):
			# print(f"The file '{save_path}' already exists. Skipping download.")
			return

	print('this could be loonnngggg .....')
	
	response = requests.get(url, stream=True)
	with open(save_path, 'wb') as f:
			for chunk in response.iter_content(chunk_size=8192):
					if chunk:
							f.write(chunk)
	
	print(f"Downloaded {url} to {save_path}")

def safe_write(path):
	ensure_directory_exists(path)
	return open(path, 'w')

def throw(message):
	print('\n')
	raise Exception(f"{message}\n\n")
