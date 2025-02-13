import os


# dir = join(".", "my-venv")
# create(dir, with_pip=True)
# os.system("Start-Process powershell -Verb runAs")
# where requirements.txt is in same dir as this script
# os.system(".\\my-venv\\Scripts\\Activate.ps1")
# os.system("pip install -r requirements.txt --user")

# import os
# import platform
# import virtualenv


# os.system("pip install -r requirements.txt --user")
from time import time
import av
import numpy as np
import supervision as sv
from PIL import Image
import cv2
import supervision as sv
from ultralytics import YOLOv10
import tensorflow as tf
import tensorflow.keras.layers as layers
import random
import math


def get_filename(fpath):
    return fpath.replace('\\', '/').rsplit('/', 1)[-1]
def get_ext(fpath):
    fname = get_filename(fpath)
    per_index = fname.index('.')
    ext = fname[per_index + 1:].lower()
    return ext
def comp(a, b, sign):
    if sign:
        return a > b
    else:
        return a < b

def check_close(detections, pixel_threshold):
    elim_list = []
    for i, det1 in enumerate(detections.xyxy):
        for j, det2 in enumerate(detections.xyxy):
            if i <= j:
                continue
            if abs(det1[0] - det2[0]) + abs(det1[1] - det2[1]) < pixel_threshold:
                elim_list.append(i)
            elif abs(det1[2] - det2[2]) + abs(det1[3] - det2[3]) < pixel_threshold:
                elim_list.append(i)
            gt = det1[0] > det1[0]
            if comp(det1[0], det2[0], gt) and comp(det1[1], det2[1], gt) and comp(det1[2], det2[2], not gt) and comp(det1[3], det2[3], not gt):
                elim_list.append(i)
    return np.unique(elim_list)


def update_det_prev(det_prev, xyxy, confidence = None, count = -1, id = None, classification = None):
    if not confidence:
        confidence = det_prev['confidence']
    if count < 0:
        count = det_prev['count']
    if not id:
        id = det_prev['id']
    if not classification:
        classification = det_prev['classification']
    return {'xyxy' : xyxy, 'confidence' : confidence, 'count' : count, 'id' : id, "classification" : classification}

def avg(a, b):
    return (a + b) / 2
def new_xyxy_f(d, p):
    return np.asarray([avg(d[0], p[0]), avg(d[1], p[1]), avg(d[2], p[2]), avg(d[3], p[3])])

def find_change_max(detections, im, model_class, ind_map):
    det_prev = None
    for cl_no, cl in enumerate(detections.data['class_name']):
            if cl == 'Trash':
                xyxy = detections.xyxy[cl_no]
                good_class, conf = get_class(im, xyxy, model_class, ind_map)
                detections.data['class_name'][cl_no] = good_class
                detections.class_id[cl_no] = len(np.unique(detections.class_id)) + 1

                det_prev = update_det_prev(None, detections.xyxy[cl_no], detections.confidence[cl_no], 0, len(np.unique(detections.class_id)) + 1, good_class)
                break 

    return detections, det_prev    
def ann_img_helper(im: np.ndarray, model, label_annotator = sv.LabelAnnotator(text_scale = 0.4, text_padding = 1), bounding_box_annotator = sv.BoxCornerAnnotator(), verbose = False, conf_level = 0.05, name_labels = True, gold_class = 'Trash', cl_id = 10, find_one = True, pixel_threshold = 20, det_prev = None, max_error = 10, dupFlag = True, trackFlag = True, model_class = None, ind_map = None) -> np.ndarray:
    results = model(im, conf=conf_level, verbose = verbose)[0]
    if name_labels:
        used_labels = np.array(results.boxes.conf.cpu())
    else:
        conf_array = np.array(results.boxes.conf.cpu())
        used_labels = [str(round(x * 100)) + '%' for x in conf_array]
    

    tot_time = sum([x for x in results.speed.values()])
    
    detections = sv.Detections.from_ultralytics(results)
    if dupFlag:
        too_close = check_close(detections, pixel_threshold=pixel_threshold)
        if len(too_close) > 1:
            too_close.sort()
            too_close = too_close[::-1]
            detections.class_id = np.delete(detections.class_id, too_close, axis = 0)
            for x in too_close:
                detections.xyxy = np.delete(np.asarray(detections.xyxy), x, 0)
            detections.confidence = np.delete(detections.confidence, too_close)
            detections.data['class_name'] = np.delete(detections.data['class_name'], too_close)
    if trackFlag:
        if not det_prev:
            detections, det_prev = find_change_max(detections, im, model_class, ind_map)
        else:
            # check closeness of top corner w/ new boxes, update if better
            found=False
            for cl_no, el in enumerate(detections.xyxy):
                if abs(el[0] - det_prev['xyxy'][0]) + abs(el[1] - det_prev['xyxy'][1]) < pixel_threshold or abs(el[2] - det_prev['xyxy'][2]) + abs(el[3] - det_prev['xyxy'][3]) < pixel_threshold:
                    found = True
                    det_prev['count'] = 0
                    detections.data['class_name'][cl_no] = det_prev['classification']
                    detections.class_id[cl_no] = det_prev['id']
                    new_xyxy = new_xyxy_f(detections.xyxy[cl_no], det_prev['xyxy'])
                    det_prev = update_det_prev(det_prev, new_xyxy, detections.confidence[cl_no], 0, det_prev['id'])
                    break
            if not found and det_prev and det_prev['count'] > max_error:
                # lock onto new object, too long since seeing it
                det_prev['count'] = 0
                detections, det_prev = find_change_max(detections, im, model_class, ind_map)
            elif not found and det_prev:
                # add previous detection into array artificially
                if len(detections.data['class_name']) == 0:
                    detections.data['class_name'] = np.asarray([det_prev['classification']])
                    detections.class_id = np.asarray([det_prev['id']])
                    detections.xyxy = np.asarray([det_prev['xyxy']])
                    detections.confidence = np.asarray([det_prev['confidence']])
                    # print(detections)
                else:
                    detections.class_id = np.insert(detections.class_id, 0, det_prev['id'])
                    detections.xyxy = np.insert(detections.xyxy, 0, det_prev['xyxy'], 0)
                    detections.confidence = np.insert(detections.confidence, 0, det_prev['confidence'])
                    detections.data['class_name'] = np.insert(detections.data['class_name'], 0, det_prev['classification'])                
                # mark that an error occurred, if too many switch target
                det_prev['count'] += 1


    im = Image.fromarray(im)
    annotated_image = bounding_box_annotator.annotate(
        scene=im, detections=detections)
    num_oysters = detections.xyxy.shape[0]
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels = used_labels if not name_labels else None)
    return np.asarray(annotated_image), num_oysters, tot_time, detections, det_prev

def get_class(im, xyxy, model_class, ind_map):
    if not model_class or not ind_map:
        new_class = 'Pick'
        prob = 0
    else:
        x_start = int(xyxy[0])
        x_end = int(xyxy[2])
        y_start = int(xyxy[1])
        y_end = int(xyxy[3])
        new_class, prob = predict_trash(im[x_start:x_end, y_start:y_end], model_class, ind_map)
    return new_class, prob


def ann_img(fname, model, conf_level = 0.05, dest = "."):
    im = Image.open(fname)
    out = ann_img_live(im, model, conf_level=conf_level)
    fname_out = get_filename(fname).rsplit(".")[0] + "_annotated." + get_ext(fname)
    Image.fromarray(out[0]).save(os.path.join(dest, fname_out))
    print(f"Annotated image saved to {fname_out}")
    return out

def ann_img_live(im: np.ndarray, model, conf_level = 0.05, det_prev = None, dupFlag = True, trackFlag = True, verbose = True):
    np_img_rsz = cv2.resize(im[:,:,::-1], (412, 412))
    out = ann_img_helper(np_img_rsz, model, conf_level = conf_level, det_prev = det_prev, dupFlag = dupFlag, trackFlag = trackFlag)
    Image.fromarray(out[0]).save("live_img.png")
    return out


def ann_video(input_vid: str, model, conf_level: float, out_location = '.', im_width = 416, im_height = 416, name_labels = True, fast_ann = False, model_class = None, ind_map = None):
    tot_oysters = 0
    
    container = av.open(input_vid)
    stream_vid = container.streams.video[0]
    fname = get_filename(input_vid)
    per_index = fname.index('.')
    ext = get_ext(input_vid)
    out_path = os.path.join(out_location, f'{fname[:per_index]}_annotated.{ext}')
    codec_name = stream_vid.codec_context.name
    fps = stream_vid.codec_context.rate


    outp = av.open(out_path, 'w')
    output_stream = outp.add_stream(codec_name, fps) # f"{int(fps*1000)}/1001"
    output_stream.width = im_width
    output_stream.height = im_height
    output_stream.pix_fmt = stream_vid.codec_context.pix_fmt



   
    start_overall = time()
    fr_diff_factor = 1
    det_prev = None
    for index, frame in enumerate(container.decode(stream_vid)):
        if index % fr_diff_factor < 1: # as close to zero as you are going to get
            start_fr = time()
            pil_img = frame.to_image()
            np_img = np.array(pil_img)
            np_img_resize = cv2.resize(np_img, (im_width, im_height))
            np_rot = np_img_resize[:, :, ::-1]
            an_mg, num_oysters, _, det, det_prev = ann_img_helper(np_rot, model, conf_level = conf_level, name_labels=name_labels, det_prev = det_prev, model_class=model_class, ind_map = ind_map)
            tot_oysters += num_oysters
            frame_out = av.VideoFrame.from_ndarray(an_mg, format='bgr24')
            end_fr = time()
            if index == 1:
                time_fr = end_fr - start_fr
                freq_fr = 1 / time_fr if fast_ann else fps
                fr_diff_factor = float(fps + 2.427243e-10) / freq_fr if fast_ann else 1
                # print(freq_fr, str(fps))
                

            if index != 0: # first frame takes longer for some reason, so discard it
                pkt = output_stream.encode(frame_out)
                outp.mux(pkt)
        
    end_overall = time()
    net_time = end_overall - start_overall

    packet = output_stream.encode(None)
    outp.mux(packet)
    outp.close()

    container.close()
    # print(os.path.exists(out_path))
    ann_rate = (index / fps) / net_time # ratio of time to annotate versus length of video
    #1 / fr_diff_factor percentage of frames annotated, will be lower if using fast_ann
    return tot_oysters * fr_diff_factor / index, net_time, out_path, ann_rate, fr_diff_factor


def run_live(model):
    det_prev = None
    while(True):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        out = ann_img_live(frame, model, conf_level = 0.5, det_prev=det_prev, dupFlag = False, trackFlag = False)
        det_prev = out[4]


def ml_compile(csi, labels, epochs=20, batch_size=128, ratio_test=1/6, ratio_validate = 1/6, complexity_scale = 3, plotting = False, return_min_acc = False, show_heatmap = True, diag_min = False):
    to_shuffle=[[csi[i], labels[i]] for i in range(len(labels))]
    random.shuffle(to_shuffle)

    shuffled_data=np.asarray([x[0] / np.max(x[0]) for x in to_shuffle])
    shuffled_labels=np.asarray([y[1] for y in to_shuffle])

    l=shuffled_data.shape[0]
    splt_test=int(l*ratio_test)
    splt_val=int(l*ratio_validate)
    data_test=np.take(shuffled_data, range(0,splt_test), axis=0)
    data_val=np.take(shuffled_data, range(splt_test,splt_test+splt_val), axis=0)
    data_train=np.take(shuffled_data, range(splt_test+splt_val,l), axis=0)
    labels_test=np.asarray(shuffled_labels[:splt_test])
    labels_val=np.asarray(shuffled_labels[splt_test:splt_test+splt_val])
    labels_train=np.asarray(shuffled_labels[splt_test+splt_val:])
    model = tf.keras.models.Sequential([
        layers.Conv2D(128, (2, 2), activation='relu', input_shape=data_train.shape[1:]),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (2, 2), activation='relu'),
        layers.Dropout(rate=.4),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(32, (2, 2), activation='relu'),
        layers.Dropout(rate=.4),
        layers.Flatten(),
        layers.Dense(128 * (2 ** complexity_scale), activation='relu'),
        layers.Dense(64 * (2 ** complexity_scale), activation='relu'),
        layers.Dense(32 * (2 ** complexity_scale), activation = 'relu'),
        layers.Dense(16 * (2 ** complexity_scale), activation = 'relu'),
        layers.Dense(max(labels) + 1)

    ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    model.compile(optimizer='adam',
                loss=loss_fn,
                metrics=['accuracy'], run_eagerly=True)
    history = model.fit(data_train,labels_train,epochs=epochs, batch_size=batch_size, validation_data=(data_val, labels_val))
    # print(data_test.shape, len(labels_test))


    model.save("models/most_recent.keras")
    print("Saved model to models/most_recent.h5")

def read_in_train(dir_name = "garbage-classification\\garbage_classification", shape_new = (50,50), max_stop = -1):
    tot_arr = []
    label_arr = []
    for label, folder in enumerate(os.listdir(dir_name)):
        print(folder)
        i = 0
        for file in os.listdir(os.path.join(dir_name, folder)):
            i = 0
            if max_stop == -1 or i < max_stop:
                i += 1
                im = Image.open(os.path.join(dir_name, folder, file))
                np_im = np.asarray(im)
                if len(np_im.shape) < 3:
                    continue
                np_im_rsz = cv2.resize(np_im, shape_new)
                Image.fromarray(np_im_rsz).save(f"scrap/{folder}_sample.png")
                tot_arr.append(np_im_rsz)
                label_arr.append(label)
    return np.asarray(tot_arr), np.asarray(label_arr)

def train_classification(dir_name = "garbage-classification/garbage_classification"):
    data, labels = read_in_train(dir_name = dir_name)
    ml_compile(data, labels)

def predict_trash(np_im, model, ind_map, input_shape = (50,50)):
    np_resize = np.asarray([cv2.resize(np_im, input_shape)])
    pred = model.predict(np_resize)[0]
    pred_sm = softmax(pred)
    # print(pred)
    return ind_map[np.argmax(pred_sm)], max(pred_sm)

def softmax(arr):
    found_max_abs = max([abs(x) for x in arr])
    arr_min = [ x / found_max_abs for x in arr]
    tot = sum([math.exp(x) for x in arr_min])
    res = [math.exp(x) / tot for x in arr_min]
    return res

def make_ind_mapping(dir_name = "garbage-classification/garbage_classification"):
    ind_map = dict()
    for label, folder in enumerate(os.listdir(dir_name)):
        ind_map[label] = folder
    return ind_map