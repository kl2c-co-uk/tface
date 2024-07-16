
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
	import hashlib
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
		import traceback
		stack = traceback.format_stack()[:-1]
		message = '\nTODO: '+message+'\n'

		for frame in stack:
			message += str(frame)
		key = md5(message)

		if not key in todo_seen:
			todo_seen[key] = True
			print(message)
