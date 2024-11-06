from objecttracker.deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
from objecttracker.deep_sort.deep_sort.detection import Detection
from objecttracker.deep_sort.deep_sort import nn_matching
import numpy as np

class Modified_Tracker:
    tracker = None
    encoder = None
    tracks = None

    def __init__(self,max_cosine_distance,min_confidence,max_iou_distance,max_age,n_init):
        # max_cosine_distance = 0.2 # Try different value, the yt video suggest to set 0.4
        nn_budget = None
        # min_confidence=0.2 # the confidence is settled in the YOLOv8 detection part as well
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = DeepSortTracker(metric,max_iou_distance, max_age, n_init) # tracker initalized

    def update(self,bboxs,confidence,feats,class_ids,stream_id):
        # for bboxs: we want xywh
        if len(bboxs) == 0:
            self.tracker.predict()
            self.tracker.update([])  
            self.update_tracks()
            return
        dets = []
        for bbox_id, bbox in enumerate(bboxs):
            dets.append(Detection(bbox, confidence[bbox_id], feats[bbox_id],class_ids[bbox_id])) 
            # 同时我也认为每一个det[i]是一个实例，每次是在调用这个detection类的下面的实例
            # add detection class into the dets list, and the variable dets is able to use all methods of Detections,e.g dets[0].to_xyah
        # print(f'dets:{dets[bbox_id].feature}')

        self.tracker.predict()
        self.tracker.update(dets,stream_id)
    #    matched_feature = self.result_match(bboxs,feats)
        self.update_tracks()

    def update_tracks(self):
        # matching_index = [index_bbox,index_detected_bbox]
        tracks = []
        for i,track in enumerate(self.tracker.tracks):
            # if not track.is_confirmed() or track.time_since_update > 1:
            #     continue
            if track.time_since_update > 1:
                continue
            # bbox = track.to_tlbr() # This value is from [mean,variance]
            bbox = track.detection_bbox[-1]
            feat = track.features[-1]
            confidence = track.confidence[-1]
            class_id = track.class_id
            age = track.age
            # print(len(track.features))
            # print(len(feat))
            # track.features = []

            id = track.track_id

            tracks.append(Track(id, bbox,feat,confidence,class_id,age))

        self.tracks = tracks

    def result_match(self,bbox,feats):
        detected_bboxs = []
        tracks = []
        matching_index = []
        matched_feature = []
        for track in self.tracker.tracks:
            detected_bboxs.append(track.to_tlbr()) # min x, min y, max x, max y :x1,y1,x2,y2

        for index_bbox, bbox in enumerate(bbox):
            for index_detected_bbox,detected_bbox in enumerate(detected_bboxs):
                if self._calculate_iou(detected_bbox,bbox)>=0.5:
                    if index_bbox not in matching_index:
                        matching_index.append(index_bbox)
                        break
        matched_feature.append(feats[index_bbox]) 
        return matched_feature

    def _calculate_iou(self,boxA, boxB):
        # Determine the coordinates of the intersection rectangle
        # print(f"boxA:{boxA}")
        # print(f"boxB:{boxB}")
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Compute the area of intersection
        interArea = max(0, xB - xA) * max(0, yB - yA)

        # Compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou

    def clear(self):
        self.tracks=[]


class Track:
    track_id = None
    bbox = None
    feat = None
    confidence = None

    def __init__(self, id, bbox,feat,confidence, class_id,age):
        self.track_id = id
        self.bbox = bbox
        self.feat = feat
        self.confidence = confidence
        self.class_id = class_id
        self.age = age