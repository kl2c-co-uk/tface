

class Invoice():
	def __init__(self):
		self._todo = {}

	@property
	def waiting(self):
		return len(self._todo)

	def request(self, key, con):
		if not key in self._todo:
			self._todo[key] = []
		self._todo[key].append(con)
	
	def respond(self, key, val):
		if key in self._todo:
			todo = self._todo[key]
			self._todo[key] = []
			for item in todo:
				item(val)


def zipTrek(path, prefix=''):
	import zipfile
	with zipfile.ZipFile(path, 'r') as file:
		for info in file.infolist():
			name = info.filename.strip()
			if name.endswith('/'):
				continue
			if not name.startswith(prefix):
				continue
			name = name[len(prefix):]

			class LoadLet:
				def __init__(self, file, info):
					self._file = file
					self._info = info
					self._data = []
					self._cold = True
				def read(self):
					if self._cold:
						self._cold = False
						self._data = self._file.read(self._info)
					return self._data


			load = LoadLet(file, info)
			yield (name, lambda:load.read())
