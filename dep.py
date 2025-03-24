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
from ultralytics import YOLO
import tensorflow as tf
import tensorflow.keras.layers as layers
import random
import math
import matplotlib.pyplot as plt
from sahi import AutoDetectionModel
from torchvision import transforms
import torch
import paramiko
from scp import SCPClient


from sahi.utils.ultralytics import (
    download_yolo11n_model, download_yolo11n_seg_model,
    # download_yolov8n_model, download_yolov8n_seg_model
)
from sahi.predict import get_prediction, get_sliced_prediction, predict
# from f5.bigip.tm import ltm


LIVE_IMAGE = "live_img.png"

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

def check_close(detections, pixel_threshold, remove_any = False): #used nested for loop which ran in n^2 time
    begin = time()
    if remove_any:
        xyxy = detections.xyxy
        n = len(xyxy)

        if n <= 1:
            return np.array([])
        
        diffs = np.abs(xyxy[:, np.newaxis, :] - xyxy[np.newaxis, : , :])

        close_x = diffs[:, :, 0] + diffs[:, :, 2] < pixel_threshold  # (n, n)
        close_y = diffs[:, :, 1] + diffs[:, :, 3] < pixel_threshold  # (n, n)
        close_overall = np.logical_or(close_x, close_y)

        gt_mask = xyxy[:, np.newaxis, :] > xyxy[np.newaxis, :, :]
        lt_mask = xyxy[:, np.newaxis, :] < xyxy[np.newaxis, :, :]

        contained = np.all(np.logical_and(gt_mask[:, :, :2], lt_mask[:, :, 2:]), axis=2)
        contained = np.logical_or(contained, np.all(np.logical_and(lt_mask[:, :, :2], gt_mask[:, :, 2:]), axis=2))

        elim_mask = np.logical_or(close_overall, contained)  # Combine close and contained checks
        elim_mask = np.triu(elim_mask, k=1)  # Consider only upper triangle (avoid duplicates and self-comparisons)

        elim_indices = np.where(np.any(elim_mask, axis=1))[0]
        # for i, x in enumerate(detections.data['class_name']):
        #     if x != "Trash":
        #         elim_indices =np.asarray(list(elim_indices) + [i])
        

        return np.unique(elim_indices)
    else:
        return []
    '''elim_list = []
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
    return np.unique(elim_list)'''


def update_det_prev(det_prev, im, xyxy, confidence = None, count = -1, id = None, classification = None, good_count = -1, midas_dist = None, midas = None, transform = None):
    if not confidence:
        confidence = det_prev['confidence']
    if count < 0:
        count = det_prev['bad_count']
    if good_count < 0:
        good_count = det_prev['good_count']
    if not id:
        id = det_prev['id']
    if not classification:
        classification = det_prev['classification']
    if not det_prev or 'history' not in det_prev.keys():
        history = [good_count]
    else:
        det_prev['history'].append(good_count)
        history = det_prev['history']
    if not det_prev or 'history_acc' not in det_prev.keys():
        history_acc = [confidence]
    else:
        det_prev['history_acc'].append(confidence)
        history_acc = det_prev['history_acc']
    if midas and transform and (not det_prev or 'midas' not in det_prev.keys()):
        midas_dist = get_midas(im, midas, xyxy, transform)
    elif not midas:
        midas_dist = -1
    return {'xyxy' : xyxy, 'confidence' : confidence, 'bad_count' : count, 'id' : id, "classification" : classification, "good_count" : good_count, "history" : history, "history_acc" : history_acc, 'midas_dist' : midas_dist}

def avg(a, b):
    return (a + b) / 2

def new_xyxy_f(d, p):
    return np.asarray([avg(d[0], p[0]), avg(d[1], p[1]), avg(d[2], p[2]), avg(d[3], p[3])])

def find_change_max(detections: sv.Detections, detection_track, im, model_class, ind_map, pick_class=10, midas = None, transform=None):
    det_prev = None
    for cl_no, cl in enumerate(detections.data['class_name']):
            if cl == 'Trash':
                xyxy = detections.xyxy[cl_no]
                # good_class, conf = get_class(im, xyxy, model_class, ind_map)
                good_class, conf = "Trash", 0
                # detections.data['class_name'][cl_no] = good_class
                
                # detections.class_id[cl_no] = np.asarray([pick_class])
                
                det_prev = update_det_prev(None, im, detections.xyxy[cl_no], detections.confidence[cl_no], 0, len(np.unique(detections.class_id)) + 1, good_class, 0, midas=midas, transform=transform)
                # detection_track.data['class_name'] = np.asarray([good_class])
                # detection_track.class_id = np.asarray([pick_class])
                # detection_track.xyxy = np.asarray([xyxy], dtype=np.float32)
                # detection_track.confidence =np.asarray([detections.confidence[cl_no]])

                break
    return detections, det_prev    

def coco_to_detections(coco):
    det = sv.Detections.empty()
    if coco:
        xyxy = []
        confidence = []
        class_name = []
        class_id = []

        for x in coco:

            xyxy_mod = [x['bbox'][0], x['bbox'][1], x['bbox'][0] + x['bbox'][2], x['bbox'][1] + x['bbox'][3]]
            xyxy.append(xyxy_mod)
            confidence.append(x['score'])
            class_name.append(x['category_name'])
            class_id.append(x['category_id'])
        det.xyxy = np.asarray(xyxy)
        det.confidence = np.asarray(confidence)
        det.data['class_name'] = np.asarray(class_name, dtype='<U15')
        det.class_id = np.asarray(class_id)
    else:
        det.data['class_name'] = np.asarray([], dtype='<U15')
    return det
def get_midas(im, midas, xyxy, transform):
    # print("Performing MIDAS depth estimation")
    img_input = transform(im).unsqueeze(0)
    with torch.no_grad():
        depth_map = midas(img_input).squeeze().cpu()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_h, depth_w = depth_map.shape
    orig_h, orig_w = im.shape[0], im.shape[1]
    x_center = (xyxy[0] + xyxy[2]) // 2
    y_center = (xyxy[1] + xyxy[3]) // 2
    x_scaled = int((x_center / orig_w) * depth_w)
    y_scaled = int((y_center / orig_h) * depth_h)
    
    return float(depth_map[y_scaled, x_scaled])
def detection_empty(detections):
    return len(detections.confidence) == 0

def add_detection(detections: sv.Detections, det_prev):

    if detection_empty(detections):
        detections.data['class_name'] = np.asarray([det_prev['classification']])
        detections.class_id = np.asarray([det_prev['id']])
        detections.xyxy = np.asarray([det_prev['xyxy']])
        detections.confidence = np.asarray([det_prev['confidence']])
    else:
        detections.class_id = np.insert(detections.class_id, 0, det_prev['id'])
        detections.xyxy = np.insert(detections.xyxy, 0, det_prev['xyxy'], 0)
        detections.confidence = np.insert(detections.confidence, 0, det_prev['confidence'])
        detections.data['class_name'] = np.insert(detections.data['class_name'], 0, det_prev['classification'])

    return detections

def add_leaner(detections: sv.Detections, detections_track: sv.Detections, det_prev, cl_no = -1):
    new_xyxy = new_xyxy_f(detections.xyxy[cl_no], det_prev['xyxy'])

    if cl_no == -1:
        
        detections['class_name'] = np.asarray([det_prev['classification'] + "?"] )
        detections.class_id = np.asarray([det_prev['id']+1])
        detections.xyxy = np.asarray([new_xyxy])
        detections.confidence = np.asarray([det_prev['confidence']])
def ann_img_helper(im: np.ndarray, model = None, model_sahi = None, label_annotator = sv.LabelAnnotator(text_scale = 1, text_thickness=2), bounding_box_annotator = sv.BoxAnnotator(thickness=10), verbose = False, conf_level = 0.05, name_labels = True, gold_class = 'Trash', cl_id = 10, find_one = True, pixel_threshold = 50, det_prev = None, max_error = 20, dupFlag = True, trackFlag = True, model_class = None, ind_map = None, print_time = True, csv_out = False, t0 = 0, good_count_threshold = 300, annotate_all = False, midas = None, transform = None) -> np.ndarray:
    persistence = 20
    persistence_gap = 20
    leaning_factor = 2
    leaning_value = persistence // leaning_factor
    start_ann = time()
    if model_sahi:
        print("Sahi model is no longer supported") if print_time else None
        # print("Using Sahi Model") if print_time else None
        # results_sahi = get_sliced_prediction(
        #     im,
        #     model_sahi,
        #     slice_height = 256,
        #     slice_width = 256,
        #     overlap_height_ratio = 0.2,
        #     overlap_width_ratio = 0.2
        # )
        # tot_time = 0
        # res_raw = results_sahi.to_coco_annotations()
        # detections = coco_to_detections(res_raw)
    if model:
        print("Using baseline YOLO Model") if print_time else None
        results = model(im, conf=conf_level, verbose = verbose)[0] # Takes 100 ms on my machine, BAD, try to get to 50

        if name_labels:
            used_labels = np.array(results.boxes.conf.cpu())
        else:
            conf_array = np.array(results.boxes.conf.cpu())
            used_labels = [str(round(x * 100)) + '%' for x in conf_array]
        tot_time = sum([x for x in results.speed.values()])
        
        detections = sv.Detections.from_ultralytics(results)
    else:
        assert("Please provide either a YOLO or Sahi model to perform analysis")
    
    end_pred = time()
    print((end_pred - start_ann) * 1000, "ms for prediction") if print_time else None 
    detection_track = sv.Detections.empty()

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
            detections, det_prev = find_change_max(detections, detection_track, im, model_class, ind_map, midas = midas, transform=transform)
        else:
            # check closeness of top corner w/ new boxes, update if better
            found=False
            for cl_no, el in enumerate(detections.xyxy):
                if abs(el[0] - det_prev['xyxy'][0]) + abs(el[1] - det_prev['xyxy'][1]) < pixel_threshold or abs(el[2] - det_prev['xyxy'][2]) + abs(el[3] - det_prev['xyxy'][3]) < pixel_threshold:
                    det_prev['good_count'] = min(det_prev['good_count']+10, persistence + persistence_gap)
                    det_prev['bad_count'] = 0
                    det_prev['confidence'] = detections.confidence[cl_no]
                    if det_prev['good_count'] > persistence:
                        found = True

                        detections.data['class_name'][cl_no] = det_prev['classification']
                        detections.class_id[cl_no] = det_prev['id']
                        new_xyxy = new_xyxy_f(detections.xyxy[cl_no], det_prev['xyxy'])

                        detection_track['class_name'] = np.asarray([det_prev['classification']])
                        detection_track.class_id = np.asarray([det_prev['id']])
                        detection_track.xyxy = np.asarray([new_xyxy])
                        detection_track.confidence = np.asarray([det_prev['confidence']])
                        det_prev = update_det_prev(det_prev, im, new_xyxy, detections.confidence[cl_no], 0, det_prev['id'], det_prev['classification'], det_prev['good_count'], midas=midas, transform=transform)
                        break
                    elif det_prev['good_count'] > leaning_value:
                        found = True
                        
                        detections.data['class_name'][cl_no] = det_prev['classification'] + "?"
                        detections.class_id[cl_no] = det_prev['id'] + 1
                        new_xyxy = new_xyxy_f(detections.xyxy[cl_no], det_prev['xyxy'])

                        detection_track['class_name'] = np.asarray([det_prev['classification'] + "?"] )
                        detection_track.class_id = np.asarray([det_prev['id']+1])
                        detection_track.xyxy = np.asarray([new_xyxy])
                        detection_track.confidence = np.asarray([det_prev['confidence']])

                        det_prev = update_det_prev(det_prev, im, new_xyxy, detections.confidence[cl_no], 0, det_prev['id'], det_prev['classification'], det_prev['good_count'], midas=midas, transform=transform)
                        break

            if not found:

                det_prev['good_count'] = max(0, det_prev['good_count'] - 1) if det_prev['good_count'] < persistence else persistence + persistence_gap
                det_prev['bad_count'] += 1
                if det_prev['bad_count'] > max_error:
                    if max(det_prev['history']) > persistence:
                        plt.plot(det_prev['history'], label="Persistence versus Time")
                        plt.xlabel("Time")
                        plt.ylabel("Persistence Score")
                        if not os.path.exists("plots"):
                            os.mkdir("plots")
                        plt.savefig("plots/persistence_plot.png")
                        plt.close()
                        # print(det_prev['history_acc'])
                        # print(det_prev['history'])
                        plt.plot(det_prev['history_acc'], label="Confidence versus Time")
                        plt.xlabel("Time")
                        plt.ylabel("Confidence")
                        if not os.path.exists("plots"):
                            os.mkdir("plots")
                        plt.savefig("plots/confidence_plot.png")
                        plt.close()
                    # print(det_prev['history'])

                    det_prev = None

            if not found and det_prev and det_prev['bad_count'] > max_error:
                # lock onto new object, too long since seeing it
                detections, det_prev = find_change_max(detections, detection_track, im, model_class, ind_map, midas = midas, transform=transform)
            elif not found and det_prev and det_prev['good_count'] > persistence:
                # add previous detection into array artificially
                detections = add_detection(detections, det_prev)
                
                detection_track = add_detection(detection_track, det_prev)


            elif not found and det_prev and det_prev['good_count'] > leaning_value:
                detection_track['class_name'] = np.asarray([det_prev['classification'] + "?"])
                detection_track.class_id = np.asarray([det_prev['id'] + 1])
                detection_track.xyxy = np.asarray([det_prev['xyxy']])
                detection_track.confidence = np.asarray([det_prev['confidence']])       
                # mark that an error occurred, if too many switch target
    if det_prev:
        det_prev['history'].append(det_prev['good_count'])
        det_prev['history_acc'].append(det_prev['confidence'])
        # print(det_prev['confidence'])
    im = Image.fromarray(im)
    num_oysters = detections.xyxy.shape[0]
    if annotate_all:
        im = bounding_box_annotator.annotate(
            scene=im, detections=detections)
        im = label_annotator.annotate(
            scene=im, detections=detections, labels = detections.data['class_name'])
    if detection_track and len(detection_track['class_name']) != 0:
        im = bounding_box_annotator.annotate(
            scene=im, detections=detection_track)
        im = label_annotator.annotate(
            scene=im, detections=detection_track, labels = detection_track.data['class_name'])
    
    end_ann = time()
    print((end_ann - start_ann) * 1000,  "ms for whole image annotation") if print_time else False
    if csv_out:
        add_csv_main(det_prev=det_prev, t0=t0)
        add_csv_verbose(detections=detections, t0=t0)
        
    return np.asarray(im), num_oysters, tot_time, detections, det_prev # takes about 400 ms total without classification, 500 ms with, bad

def get_class(im, xyxy, model_class, ind_map):
    if not model_class or not ind_map:
        new_class = 'Pick'
        prob = 0
    else:
        x_start = int(xyxy[0])
        x_end = int(xyxy[2])
        y_start = int(xyxy[1])
        y_end = int(xyxy[3])
        new_class, prob = predict_trash(im[y_start:y_end, x_start:x_end], model_class, ind_map)
        # new_class, prob = "Pick", 0
    
    return new_class, prob


def load_sahi(path_sahi, conf_level):
    # if path_sahi:
    #     model_sahi = AutoDetectionModel.from_pretrained(model_type='yolov11', model_path=path_sahi, confidence_threshold = conf_level, device = "cpu")
    #     return model_sahi
    # else:
    return None

def ann_img(fname, model, path_sahi = None, conf_level = 0.05, dest = ".",csv_out=False, t0 = 0, label_annotator = sv.LabelAnnotator(text_scale = 1, text_thickness=2), bounding_box_annotator = sv.BoxAnnotator(thickness=3), out_size = (412, 412), annotate_all = True, det_prev = None, save_photo = True):
    
    model_sahi = load_sahi(path_sahi, conf_level)
    
    im = Image.open(fname)
    out = ann_img_live(im, model, conf_level=conf_level, model_sahi = model_sahi, csv_out=csv_out, t0 = t0, label_annotator=label_annotator, out_size=out_size, bounding_box_annotator = bounding_box_annotator, annotate_all = annotate_all, det_prev=det_prev, save_photo = False)
    if save_photo:
        fname_out = get_filename(fname).rsplit(".")[0] + "_annotated." + get_ext(fname)
        Image.fromarray(out[0]).save(os.path.join(dest, fname_out))
        print(f"Annotated image saved to {fname_out}")
    return out, out[4]

def ann_img_live(im: np.ndarray, model, model_sahi = None, conf_level = 0.05, det_prev = None, dupFlag = True, trackFlag = True, verbose = True, max_error = 10, csv_out=False, t0 = 0,  label_annotator = sv.LabelAnnotator(text_scale = 0.4, text_padding = 1), out_size = (412, 412), bounding_box_annotator = sv.BoxAnnotator(thickness=1), annotate_all = False, save_photo = True):
    im_np = np.array(im)
    # np_img_rsz = cv2.resize(im_np[:,:,::-1], out_size)
    np_img_rsz = im_np
    out = ann_img_helper(np_img_rsz, model, model_sahi = model_sahi, conf_level = conf_level, det_prev = det_prev, dupFlag = dupFlag, trackFlag = trackFlag, max_error = max_error, csv_out=csv_out, t0 = t0, label_annotator=label_annotator, bounding_box_annotator = bounding_box_annotator, annotate_all = annotate_all)
    if save_photo:
        Image.fromarray(out[0]).save(LIVE_IMAGE)
    return out


def ann_video(input_vid: str, model, conf_level: float, sahi_path = None, out_location = '.', im_width = 416, im_height = 416, name_labels = True, fast_ann = False, model_class = None, ind_map = None, bbox_thickness=3, text_scale=1, text_thickness=2, compress = True, max_error=5, csv_out=False):
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
    output_stream.width = stream_vid.width if not compress else im_width
    output_stream.height = stream_vid.height if not compress else im_height
    output_stream.pix_fmt = stream_vid.codec_context.pix_fmt
    print_warning = True
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
    midas.eval()
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((384, 384)),  # Resize for MiDaS model
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    tm_csv = init_csv()

   
    start_overall = time()
    fr_diff_factor = 1
    det_prev = None
    model_sahi = load_sahi(sahi_path, conf_level)


    for index, frame in enumerate(container.decode(stream_vid)):
        if index % fr_diff_factor < 1: # as close to zero as you are going to get
            start_fr = time()
            pil_img = frame.to_image()
            np_img = np.array(pil_img)
            try:
                np_img_resize = np_img if not compress else cv2.resize(np_img, (im_width, im_height))
            except Exception:
                if print_warning:
                    print_warning=False
                    print("Image is too small for specified width and height, using uncompressed frames")
                np_img_resize = np_img
            np_rot = np_img_resize[:, :, ::-1]
            an_mg, num_oysters, _, det, det_prev = ann_img_helper(np_rot, model, model_sahi = model_sahi, conf_level = conf_level, name_labels=name_labels, det_prev = det_prev, model_class=model_class, ind_map = ind_map, print_time = index == 1, t0 = tm_csv, bounding_box_annotator = sv.BoxAnnotator(thickness=bbox_thickness), label_annotator=sv.LabelAnnotator(text_scale=text_scale, text_thickness=text_thickness), max_error=max_error, csv_out=csv_out, midas = midas, transform=transform)
            # Image.fromarray(an_mg).save("live_img.png")
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
    cv2.destroyAllWindows()
    container.close()
    # print(os.path.exists(out_path))
    ann_rate = (index / fps) / net_time # ratio of time to annotate versus length of video
    #1 / fr_diff_factor percentage of frames annotated, will be lower if using fast_ann
    
    return tot_oysters * fr_diff_factor / index, net_time, out_path, ann_rate, fr_diff_factor
def init_csv(filename_main = "tracks/output_pick.csv", filename_verbose="tracks/output_verbose.csv"):
    if not os.path.exists("tracks/"):
        os.mkdir("tracks/")
    with open(filename_main, "w") as f:
        f.write("X,Y,Confidence,Object,Time (MS),Persistence,Bad Count,MiDAS Scale\n")
    with open(filename_verbose, "w") as f:
        f.write("X,Y,Confidence,Object,Time (MS)\n")
    return time()

def add_csv_main(filename = "tracks/output_pick.csv", det_prev = None, t0 = 0):
    try:
        if det_prev:
            with open(filename, "a") as f:
                xyxy = det_prev['xyxy']
                f.write(f"{round(avg(xyxy[0], xyxy[2]))},{round(avg(xyxy[1], xyxy[3]))},{round(det_prev['confidence'] * 100)},{det_prev['classification']},{int((time() - t0) * 1000)},{det_prev['good_count']},{det_prev['bad_count']},{det_prev['midas_dist']}\n")
    except Exception:
        print("Couldn't write, permission denied")
def add_csv_verbose(filename ="tracks/output_verbose.csv", detections: sv.Detections = None, t0 = 0):
    try:
        if len(detections["class_name"]) != 0:
            with open(filename, "a") as f:
                for ind, itm in enumerate(detections["class_name"]):
                    pos = detections.xyxy[ind]
                    f.write(f"{round(avg(pos[0], pos[2]))},{round(avg(pos[1], pos[3]))},{round(detections.confidence[ind] * 100)},{detections.data['class_name'][ind]},{int((time() - t0) * 1000)}\n")
    except Exception:
        print("Couldn't write, permission error")


def run_live(model, sahi_path = None, conf = 0.5, max_error = 3,csv_out=True):
    det_prev = None
    tm = init_csv()
    cap = cv2.VideoCapture(0)
    model_sahi = load_sahi(sahi_path, conf)
    while(True):
        ret, frame = cap.read()
        if not ret:
            assert("Error in video capture. Please check your hardware.")
        out = ann_img_live(frame, model, model_sahi, conf_level = conf, det_prev=det_prev, dupFlag = False, trackFlag = True, max_error = max_error, csv_out=csv_out, t0 = tm)
        det_prev = out[4]
    cv2.destroyAllWindows()

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
    # takes 100 ms
    np_resize = np.asarray([cv2.resize(np_im, input_shape)])
    time_start = time()
    pred = model.predict(np_resize)[0]
    time_end = time()
    print((time_end - time_start) * 1000, "ms for classification")
    pred_sm = softmax(pred)
    Image.fromarray(np_resize[0]).save("cur_trash.png")
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

def createSSHClient(server, port, user, password):
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, port, user, password)
    return client
def run_push_pull(server):

    port=22
    user="elytra"
    password="Elytra@2025"

    ssh = createSSHClient(server, port, user, password)

    cl = SCPClient(ssh.get_transport())
    det_prev = None
    tm_csv = init_csv()
    while(True):
        t_get = time()
        cl.get("/home/elytra/AI/cur.jpg", "test_scp/")
        t_end = time()
        print("TIME TO PULL IMAGE", t_end - t_get)
        t_start_ann = time()
        _, det_prev = ann_img("test_scp/cur.jpg", model=model_YOLO, det_prev=det_prev, csv_out=True, save_photo=False, t0 =tm_csv)
        t_end_ann = time()
        print("TIME TO ANNOTATE IMAGE", t_end_ann - t_start_ann)
        t_start_csv = time()
        cl.put("tracks/output_pick.csv", "/home/elytra/AI/")
        t_end_csv = time()
        print("TIME TO PUSH CSV", t_end_csv - t_start_csv)