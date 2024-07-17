
from datasource import todo

INPUT_WIDTH = 1920
INPUT_HEIGHT = 1080

HEATMAP_SCALE = 0.1
HEATMAP_WIDTH = int(HEATMAP_SCALE * INPUT_WIDTH)
HEATMAP_HEIGHT = int(HEATMAP_SCALE * INPUT_HEIGHT)




EPOCHS = 4
BATCH_SIZE = 1

# limit how big the datasets are durring development
LIMIT = 10

# skip faces smaller than this in their source pixels
MIN_WIDTH = 79
MIN_HEIGHT = 79
MIN_SIZE = int(MIN_HEIGHT * MIN_WIDTH * 1.1)


# how many faces are in the boundin boxx/
PATCH_COUNT = 10


# build extra images 
PERMUTE_FACES = False


# adjust the model

RESNET_TRAIN = False
RESNET_TOP = False


