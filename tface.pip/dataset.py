
###
# if you change these you'll have to delete the images and rebuild them ... that will be slow (sorry)

target_width = 1920
target_height = 1080
heatmap_scale = 0.025
target_root = 'target/'
target_images = 'target/mega-wipder-data/'

from datasource.wider import main as wider_main
from datasource.context import Cache

if __name__ == '__main__':
	cache = Cache('target/', (target_width,target_height), heatmap_scale)
	wider_main(cache, target_images)



throw('???')













