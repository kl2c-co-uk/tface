
from datasource import todo

INPUT_SIZE = 640

# small training limits for development time
# how many items per dataset to extract
LIMIT = 8

# how many times to go over everything
EPOCHS = 2


BATCH_SIZE = 8 # small to fit in the k620
# ... bigger sizes are probably fine there too ...
# 16 = 3.68G
# 12 = 3G?
# ... bugger batches finish epochs faster, but, need more GPU memeory ...


# oh look; we can change the value for the export?
EXPORT_BATCH_SIZE = 1



"""

	(the rest is out of date)

"""



INPUT_WIDTH = 1920
INPUT_HEIGHT = 1080

HEATMAP_SCALE = 0.1
HEATMAP_WIDTH = int(HEATMAP_SCALE * INPUT_WIDTH)
HEATMAP_HEIGHT = int(HEATMAP_SCALE * INPUT_HEIGHT)




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


