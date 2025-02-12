from dep import *
model_YOLO = YOLOv10('models/best.pt')
# pass in video - saves to VIDEO_NAME_annotated.mp4

##############################################
###### UPDATE WITH YOUR VIDEO PATH ###########
##############################################

# VIDEO_NAME = "media/trash v2.mp4"
VIDEO_NAME = "media/video_NYC.mp4"

##############################################

ann_video(VIDEO_NAME, model_YOLO, .4, out_location = "media")

# pass in image

# IMAGE_NAME = "media/NYC_1.jpg"
# out = ann_img(IMAGE_NAME, model_YOLO, conf_level=.2, dest = "media")
