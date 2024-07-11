
target_width = 1920
target_height = 1080
heatmap_scale = 0.1

heatmap_width = int(heatmap_scale * target_width)
heatmap_height = int(heatmap_scale * target_height)

EPOCHS = 2
BATCH_SIZE = 4

SMALL_LIMIT = 20

RESNET_TRAIN = False
RESNET_TOP = False

def sizes():
	return (
		target_width,
		target_height,
		heatmap_width,
		heatmap_height,
	)


