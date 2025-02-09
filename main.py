from dep import *
model_2 = YOLOv10('best.pt')
# pass in video - saves to VIDEO_NAME_annotated.mp4

##############################################
###### UPDATE WITH YOUR VIDEO PATH ###########
##############################################

VIDEO_NAME = "trash v2.mp4"

##############################################

ann_video_helper(VIDEO_NAME, model_2, .1)

# pass in image