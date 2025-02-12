from dep import *
model_YOLO = YOLOv10('models/best.pt')
# pass in video - saves to VIDEO_NAME_annotated.mp4

##############################################
###### UPDATE WITH YOUR VIDEO PATH ###########
##############################################

# VIDEO_NAME = "media/trash v2.mp4"
VIDEO_NAME = "media/video_NYC.mp4"

##############################################

# ann_video(VIDEO_NAME, model_YOLO, .7, out_location = "media")

# pass in image

# IMAGE_NAME = "media/NYC_1.jpg"
# out = ann_img(IMAGE_NAME, model_YOLO, conf_level=.2, dest = "media")
def run_live():
    det_prev = None
    while(True):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        im = Image.fromarray(frame[:,:,::-1])
        out = ann_img_live(im, model_YOLO, conf_level = 0.5, det_prev=det_prev, dupFlag = False, trackFlag = False)
        det_prev = out[4]
run_live()