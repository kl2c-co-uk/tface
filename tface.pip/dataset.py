
###
# if you change these you'll have to delete the images and rebuild them ... that will be slow (sorry)


from datasource.wider import main as wider_main
from datasource.context import Cache

def dataset_main():
	target_width = 1920
	target_height = 1080
	heatmap_scale = 0.025
	target_root = 'target/'
	target_images = 'target/mega-wipder-data/'

	cache = Cache('target/', (target_width,target_height), heatmap_scale)
	
	print('loading datasources ...')

	wider_main(cache, target_images)

	print('... datasources available')

	return target_images

if __name__ == '__main__':
	dataset_main()

