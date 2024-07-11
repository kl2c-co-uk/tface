
###
# if you change these you'll have to delete the images and rebuild them ... that will be slow (sorry)


from datasource.wider import main as wider_main
from datasource.context import Cache

target_width = 1920
target_height = 1080
heatmap_scale = 0.1

def sizes():
	return (
		target_width,
		target_height,
		int(heatmap_scale * target_width),
		int(heatmap_scale * target_height)
	)

def dataset_main():
	target_root = 'target/'
	target_images = 'target/mega-wipder-data/'

	cache = Cache('target/', (target_width,target_height), heatmap_scale)
	
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

