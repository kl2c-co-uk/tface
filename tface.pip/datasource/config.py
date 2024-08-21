
from datasource import todo, Blurb

#
#
# next one
# python yolo5.py --weights "C:/Users/peter/Desktop/tface-train/tface.pip/target/yolov5/runs/train/exp11/weights/best.pt" && shutdown /s /t 180
# 

# python yolo5.py --weights "C:/Users/peter/Desktop/tface-train/tface.pip/target/yolov5/runs/train/exp12/weights/best.pt" && shutdown /s /t 180
# 20 epoch
# 128 input size for better accuracy
# 72 batch size to avoid crashes - based on napkin-math; this should stay under 2G


INPUT_SIZE = 192
BATCH_SIZE = 48
EPOCHS = 2 # 40 was recomended



# 640 (or 320?) was the default
# RTFM; -1 batch size handles it automatically ... but also that crashed my PC a lot


DATASETS = Blurb(
    WIDER_train = True,
    WIDER_val = True,
    iCartoon = False,
)


# small training limits for development time
# how many items per dataset to extract
LIMIT = 0



GREENLIST_DEFAULT_INCLUE = True # if an image isn't in the green/red list - do you want it included anyway?
AUDIT = False # do you want to audit the/some cartoon images?

PREVIEW = False #0 != LIMIT





# oh look; we can change the value for the export?
EXPORT_BATCH_SIZE = 1
