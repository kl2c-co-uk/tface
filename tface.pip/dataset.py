

from datasource.base import Cache



def yset_training(cache):
	points = 0
	import datasource.config 

	import datasource.wider as wider
	import datasource.data as data
	for frame in wider.trainingFrames(cache):

		# yield the full point
		yield data.DataPoint(cache, frame)
		points += 1

		# do any perturbations of the point

		if datasource.config.LIMIT > 0 and datasource.config.LIMIT < points:
			return

def yset_validate(cache):
	points = 0
	import datasource.config 

	import datasource.wider as wider
	import datasource.data as data
	for frame in wider.validateFrames(cache):

		# yield the full point
		yield data.DataPoint(cache, frame)
		points += 1

		# do any perturbations of the point

		if datasource.config.LIMIT > 0 and datasource.config.LIMIT < points:
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

