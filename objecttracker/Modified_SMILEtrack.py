from objecttracker.SMILEtrack.SMILEtrack_Official.tracker.mc_SMILEtrack import SMILEtrack
# from objecttracker.SMILEtrack.SMILEtrack_Official.yolov7.utils.torch_utils import select_device
import argparse
# import torch
import numpy as np

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