from dep import *
from opendatasets import download
import lap

CLASSIFICATION_WEB_NAME = "https://www.kaggle.com/datasets/jasonirie/dataset-for-garbage"
CLASSIFICATION_DATA= CLASSIFICATION_WEB_NAME.rsplit('/')[5]
if not os.path.exists(CLASSIFICATION_DATA + "/"):
    print("Here", CLASSIFICATION_DATA)
    download(CLASSIFICATION_WEB_NAME)
    with open(".gitignore", "a") as f:
        f.write("\n" + CLASSIFICATION_DATA)
MAIN_MODEL_PATH = 'models/best_v11.onnx'
    

model_YOLO = YOLO(MAIN_MODEL_PATH, task='detect')
# model_YOLO.export(format="onnx")
# model_YOLO.export(format='tflite')


# model_YOLO = YOLO('models/best_v11.onnx', task="detect")
# model_tf = tf.keras.models.load_model("models/most_recent.keras")
ind_map = None
# ind_map = make_ind_mapping()
# ind_map = {0:"battery",1:"biological",2:"brown-glass",3:"cardboard",4:"clothes",5:"green-glass",6:"metal",7:"paper"}

# pass in video - saves to VIDEO_NAME_annotated.mp4

##############################################
###### UPDATE WITH YOUR VIDEO PATH ###########
##############################################

VIDEO_NAME = "media/middle.mp4"
# VIDEO_NAME = "media/middle.mp4"

##############################################
model_YOLO.predict("media/video_NYC.mp4", save=True)
# ann_video(VIDEO_NAME, model = model_YOLO, sahi_path=None, conf_level= 0, out_location = "media",model_class = None, ind_map = ind_map, bbox_thickness=3, text_scale=1, text_thickness=2, compress=False, max_error=10, csv_out=True)

# pass in image

# IMAGE_NAME = "media/bottles.jpg"
# for i in range(1,2):
#     out = ann_img("media/bottles_" + str(i) + ".jpg", model_YOLO, path_sahi=None, conf_level=.4, dest = "media", label_annotator = sv.LabelAnnotator(text_scale = 0.4, text_thickness=1, text_padding = 1), bounding_box_annotator = sv.BoxAnnotator(thickness=1))

##################### live annotation #####################

# run_live(model_YOLO, conf = 0.00003, max_error = 3)

############################################################

#### training classification - look at link under README.md if errors ####

# train_classification(CLASSIFICATION_DIR_NAME)
# ind_map = make_ind_mapping(CLASSIFICATION_DIR_NAME)
# np_im = np.asarray(Image.open("classification/garbage_classification/battery/battery50.jpg"))
# vals = predict_trash(np_im, model_tf, ind_map)
# print(vals)


##########################################################################

############## from PI (Live Annotation, Modified) ###############################
server_pi = "100.79.182.84"


run_push_pull(server_pi)
