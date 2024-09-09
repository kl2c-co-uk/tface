
from datasource import todo, Blurb

#


# thrusday; python yolo5.py && shutdown /s /t 180


# wednesday; python yolo5.py --weights "G:/My Drive/kl2c/archived-configurations/2024-08-30.that-one-thar-ran-until-noon.pt" && shutdown /s /t 180


#
# retrin the no-cartoon one with cartoons ... sorry
#  python yolo5.py --weights "C:/Users/peter/Desktop/tface-train-nocartoon/tface.pip/target/yolov5/runs/train/exp5/weights/best.pt" && shutdown /s /t 180
# 
# 20 epoch
# 128 input size for better accuracy
# 72 batch size to avoid crashes - based on napkin-math; this should stay under 2G


# 192x48 = 1.3G (in the first ones)
# 192x64 = 1.7G on epoc 1/39 so i assume it'll stay under 3G

INPUT_SIZE = 192
BATCH_SIZE = 50

EPOCHS = 80 # 40 was recomended. 80 for 



# 640 (or 320?) was the default
# RTFM; -1 batch size handles it automatically ... but also that crashed my PC a lot


DATASETS = Blurb(
    WIDER_train = True,
    WIDER_val = True,
    iCartoon = True,
)


# small training limits for development time
# how many items per dataset to extract
LIMIT = 0



GREENLIST_DEFAULT_INCLUE = False # if an image isn't in the green/red list - do you want it included anyway?
PREVIEW = False #0 != LIMIT





# oh look; we can change the value for the export?
EXPORT_BATCH_SIZE = 1
