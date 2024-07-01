
import os
import hashlib

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
