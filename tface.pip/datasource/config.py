
from datasource import todo


input_width = 1920
input_height = 1080
heatmap_scale = 0.1

todo('this should not be named target - it is the input')
target_width, target_height = input_width, input_height

heatmap_width = int(heatmap_scale * target_width)
heatmap_height = int(heatmap_scale * target_height)


EPOCHS = 4
BATCH_SIZE = 1

# limit how big the batches are durring development
LIMIT = -1

# skip faces smaller than this
MIN_WIDTH = 79
MIN_HEIGHT = 79
MIN_SIZE = int(MIN_HEIGHT * MIN_WIDTH * 1.1)


RESNET_TRAIN = False
RESNET_TOP = False


