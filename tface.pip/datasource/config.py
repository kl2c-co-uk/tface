
target_width = 1920
target_height = 1080
heatmap_scale = 0.1

EPOCHS = 2
BATCH_SIZE = 1

# limit how big the batches are durring development
LIMIT = 4

# skip faces smaller than this
MIN_WIDTH = 79
MIN_HEIGHT = 79
MIN_SIZE = int(MIN_HEIGHT * MIN_WIDTH * 1.1)


RESNET_TRAIN = False
RESNET_TOP = False


