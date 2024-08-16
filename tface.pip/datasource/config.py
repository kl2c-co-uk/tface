
from datasource import todo

INPUT_SIZE = 160 # small to make training fast

# small training limits for development time
# how many items per dataset to extract
LIMIT = 10

PREVIEW = 0 != LIMIT

# how many times to go over everything
EPOCHS = 40 # this was teh recomended



# RTFM; -1 batch size handles it automatically
BATCH_SIZE = -1


# small enough to fit in the k620



# ... bigger sizes are probably fine there too ...
# 16 = 3.68G
# 12 = 3G?
# ... bugger batches finish epochs faster, but, need more GPU memeory ...


# oh look; we can change the value for the export?
EXPORT_BATCH_SIZE = 1
