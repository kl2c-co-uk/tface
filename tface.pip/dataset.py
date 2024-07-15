
###
# if you change these you'll have to delete the images and rebuild them ... that will be slow (sorry)





from datasource.wider import main as wider_main
from datasource.context import Cache



def dataset_main():
	target_root = 'target/'
	target_images = 'target/mega-wipder-data/'

	cache = Cache('target/')
	
	print('loading datasources ...')

	wider_main(cache, target_images)

	print('... datasources available')

	return target_images

def contents():
	dataset_norms = dataset_main()
	assert dataset_norms.endswith('/')
	train_image_dir = dataset_norms + 'train/images'
	train_mask_dir = dataset_norms + 'train/masks'
	validation_image_dir = dataset_norms + 'validation/images'
	validation_mask_dir = dataset_norms + 'validation/masks'
	return (train_image_dir, train_mask_dir, validation_image_dir, validation_mask_dir)

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

