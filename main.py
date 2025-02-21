from dep import *
import opendatasets as od
CLASSIFICATION_DATA= "classification"
if not os.path.exists('garbage-classification'):
    od.download("https://www.kaggle.com/datasets/feyzazkefe/trashnet")

model_YOLO = YOLOv10('models/best.pt')
model_tf = tf.keras.models.load_model("models/most_recent.keras")
# ind_map = make_ind_mapping()
ind_map = {0:"battery",1:"biological",2:"brown-glass",3:"cardboard",4:"clothes",5:"green-glass",6:"metal",7:"paper"}

# pass in video - saves to VIDEO_NAME_annotated.mp4

##############################################
###### UPDATE WITH YOUR VIDEO PATH ###########
##############################################

VIDEO_NAME = "media/trash v2.mp4"
# VIDEO_NAME = "media/middle.mp4"

##############################################

# ann_video(VIDEO_NAME, model_YOLO, .4, out_location = "media",model_class = model_tf, ind_map = ind_map)

# pass in image

# IMAGE_NAME = "media/NYC_1.jpg"
# out = ann_img(IMAGE_NAME, model_YOLO, conf_level=.2, dest = "media")
##################### live annotation #####################

run_live(model_YOLO)

############################################################

#### training classification - look at link under README.md if errors ####

# train_classification()
# ind_map = make_ind_mapping()
# np_im = np.asarray(Image.open("classification/garbage_classification/battery/battery50.jpg"))
# vals = predict_trash(np_im, model_tf, ind_map)
# print(vals)


##########################################################################

