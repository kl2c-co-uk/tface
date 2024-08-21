
from datasource import todo

#
#
# next one
# python yolo5.py --weights "C:/Users/peter/Desktop/tface-train/tface.pip/target/yolov5/runs/train/exp11/weights/best.pt" && shutdown /s /t 180
# 

# python yolo5.py --weights "C:/Users/peter/Desktop/tface-train/tface.pip/target/yolov5/runs/train/exp12/weights/best.pt" && shutdown /s /t 180
# 20 epoch
# 128 input size for better accuracy
# 128 batch size to avoid crashes

INPUT_SIZE = 96 # very small to make training fast
BATCH_SIZE = 192

# 96 and 16 = 0.24G
# 96 and 50 = 0.42G
# 96 and 128 = 0.814G
# 96 and 256 =? =2.52G
# 96 and 192 =1.2G     <-- seems to be best match

# 640 (or 320?) was the default
# RTFM; -1 batch size handles it automatically




# small training limits for development time
# how many items per dataset to extract
LIMIT = 0



GREENLIST_DEFAULT_INCLUE = True # if an image isn't in the green/red list - do you want it included anyway?
AUDIT = False # do you want to audit the/some cartoon images?

PREVIEW = False #0 != LIMIT

# how many times to go over everything
EPOCHS = 5 # 40 was recomended








# small enough to fit in the k620



# ... bigger sizes are probably fine there too ...
# 16 = 3.68G
# 12 = 3G?
# ... bugger batches finish epochs faster, but, need more GPU memeory ...


# oh look; we can change the value for the export?
EXPORT_BATCH_SIZE = 1
