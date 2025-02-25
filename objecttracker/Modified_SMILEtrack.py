from objecttracker.SMILEtrack.SMILEtrack_Official.tracker.mc_SMILEtrack import SMILEtrack
from objecttracker.SMILEtrack.SMILEtrack_Official.yolov7.utils.torch_utils import select_device
import argparse
import torch
import numpy as np


# parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
# parser.add_argument('--weights', nargs='+', type=str, default='yolov7-e6e.pt', help='model.pt path(s)')
# parser.add_argument("--benchmark", dest="benchmark", type=str, default='MOT17', help="benchmark to evaluate: MOT17 | MOT20")
# parser.add_argument("--eval", dest="split_to_eval", type=str, default='test', help="split to evaluate: train | val | test")
# parser.add_argument('--txt-dir', dest="txt_dir", type=str, default='yolov7_track_results', help='dir to save result txt files')

# parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
# parser.add_argument('--conf-thres', type=float, default=0.2, help='object confidence threshold')
# parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
# parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
# parser.add_argument('--view-img', action='store_true', help='display results')

# parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
# parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
# parser.add_argument('--augment', action='store_true', help='augmented inference')
# parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", help="Adopting mix precision evaluating.")
# parser.add_argument("--fuse", dest="fuse", default=False, action="store_true", help="Fuse conv and bn for testing.")

# parser.add_argument('--project', default='runs/track', help='save results to project/name')
# parser.add_argument('--name', default='exp', help='save results to project/name')
# parser.add_argument('--trace', action='store_true', help='trace model')
# parser.add_argument('--hide-labels-name', default=False, action='store_true', help='hide labels')

# parser.add_argument("--default-parameters", dest="default_parameters", default=False, action="store_true", help="use the default parameters as in the paper")
# parser.add_argument("--save-frames", dest="save_frames", default=False, action="store_true", help="save sequences with tracks.")

# # tracking args
# parser.add_argument("--track_high_thresh", type=float, default=0.5, help="tracking confidence threshold")
# parser.add_argument("--track_low_thresh", default=0.45, type=float, help="lowest detection threshold")
# parser.add_argument("--new_track_thresh", default=0.6, type=float, help="new track thresh")
# parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
# parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
# parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
#                     help="threshold for filtering out boxes of which aspect ratio are above the given value.")
# parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
# parser.add_argument("--fuse-score", dest="mot20", default=False, action="store_true",
#                     help="fuse score and iou for association")

# # CMC
# parser.add_argument("--cmc-method", default="file", type=str, help="cmc method: files (Vidstab GMC) | sparseOptFlow |orb | ecc")
# parser.add_argument("--ablation", dest="ablation", default=False, action="store_true", help="ablation ")

# # ReID
# parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="with ReID module.")
# parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml",
#                     type=str, help="reid config file path")
# parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth",
#                     type=str, help="reid config file path")
# parser.add_argument('--proximity_thresh', type=float, default=0.5,
#                     help='threshold for rejecting low overlap reid matches')
# parser.add_argument('--appearance_thresh', type=float, default=0.25,
#                     help='threshold for rejecting low appearance similarity reid matches')



class Modified_Tracker:
    tracker = None
    encoder = None
    tracks = None

    def __init__(self,track_low_thresh=0.45,track_high_thresh=0.5,new_track_thresh=0.6,track_buffer=30,proximity_thresh=0.5,appearance_thresh=0.25,with_reid=False,match_thresh=0.8,frame_rate=10):
        parser = argparse.ArgumentParser()
        opt = parser.parse_args()
        opt.jde = False
        opt.exist_ok = True
        opt.device = ' '
        opt.track_low_thresh = track_low_thresh
        opt.track_high_thresh = track_high_thresh
        opt.new_track_thresh = new_track_thresh
        opt.track_buffer = track_buffer
        opt.proximity_thresh = proximity_thresh
        opt.appearance_thresh = appearance_thresh
        opt.with_reid = False
        opt.match_thresh = match_thresh

        self.tracker = SMILEtrack(opt,frame_rate) # tracker initalized

    def update(self,bboxs,confidence,feats,class_ids,img,det_array):

        if len(bboxs) == 0:
           # self.tracker.predict() SIMLEtrack do not have .predict() function
            self.tracker.update([],img)  
            self.update_tracks([])
            return
        
        detections = []

        for bbox_id, bbox in enumerate(bboxs):
            
            x, y, w, h= bbox
            x_min,y_min = x,y
            x_max, y_max = x + w, y + h
            # Create a new array for each detection
            lat,lon = det_array[bbox_id,6], det_array[bbox_id,7]
            det = np.array([x_min, y_min, x_max, y_max, confidence[bbox_id], class_ids[bbox_id],lat,lon])
            # Append the detection and the features
            if len(feats[bbox_id]) != 2048: #the recheck of feats dimension
                print('??? feature dimension is not equal to 2048x1')
            
            det = np.concatenate((det, feats[bbox_id]))
            detections.append(det)

        # Convert the list to a NumPy array
        detections = np.array(detections)

       # self.tracker.predict()
        temp_results = self.tracker.update(detections,img)
    #    matched_feature = self.result_match(bboxs,feats)
        self.update_tracks(temp_results)

    def update_tracks(self,temp_results):
        # matching_index = [index_bbox,index_detected_bbox]
        tracks = []
        for temp_res in temp_results:

            bbox = temp_res.tlwh
            features_deque = temp_res.features
            class_id = temp_res.cls
            lat,lon = temp_res.lat, temp_res.lon
            
            if len(features_deque) > 0:
                
                features_array = np.concatenate(features_deque)
                features_array = features_array[-2048:]
     
                feat = features_array.flatten()
                confidence = temp_res.score
                # print(feat.shape)
            else:
                # print("No features available for this track.")
                feat = np.zeros(2048)  # or handle as appropriate for your application
                confidence = 0 
            
            # print(f'{len(temp_results)} tracks in this frame'
            id = temp_res.track_id

            tracks.append(Track(id, bbox,feat,confidence,class_id,lat,lon))

        self.tracks = tracks

class Track:
    track_id = None
    bbox = None
    feat = None
    confidence = None
    class_id = None
    lat = None
    lon = None

    def __init__(self, id, bbox, feat, confidence, class_id, lat, lon):
        self.track_id = id
        self.bbox = bbox
        self.feat = feat
        self.confidence = confidence
        self.class_id = class_id
        self.lat = lat
        self.lon = lon