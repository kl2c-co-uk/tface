
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



if __name__ == '__main__':
	dataset_main()

