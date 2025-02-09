from dep import *
model_2 = YOLOv10('best.pt')
ann_video_helper("trash_test.mp4", model_2, .1)