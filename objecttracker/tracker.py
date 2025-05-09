import logging
import objecttracker.Modified_Deepsort
import time
import uuid
from pathlib import Path
from typing import Any
from typing import List, Tuple
import numpy as np
import cv2
import objecttracker.Modified_SMILEtrack
import torch
# from boxmot import OCSORT, DeepOCSORT
from prometheus_client import Counter, Histogram, Summary
from visionlib.pipeline.tools import get_raw_frame_data

from .config import ObjectTrackerConfig, TrackingAlgorithm

from visionapi_yq.messages_pb2 import SaeMessage, TrackletsByCamera,Trajectory,Tracklet

logging.basicConfig(format='%(asctime)s %(name)-15s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)

GET_DURATION = Histogram('object_tracker_get_duration', 'The time it takes to deserialize the proto until returning the detection result as a serialized proto',
                         buckets=(0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25))

MODEL_DURATION = Summary('object_tracker_tracker_update_duration', 'How long the tracker update takes')
OBJECT_COUNTER = Counter('object_tracker_object_counter', 'How many objects have been tracked')
PROTO_SERIALIZATION_DURATION = Summary('object_tracker_proto_serialization_duration', 'The time it takes to create a serialized output proto')
PROTO_DESERIALIZATION_DURATION = Summary('object_tracker_proto_deserialization_duration', 'The time it takes to deserialize an input proto')


class Tracker:
    def __init__(self, config: ObjectTrackerConfig) -> None:
        self.config = config
        logger.setLevel(self.config.log_level.value)
        self.object_id_seed = uuid.uuid4()
        self._setup()
        self.height = None
        self.width = None
        
    def __call__(self, input_proto, *args, **kwargs) -> Any:
        return self.get(input_proto)

    @GET_DURATION.time()
    @torch.no_grad()
    def get(self, input_proto,stream_id) -> List[Tuple[str, bytes]]:       

        input_image, sae_msg = self._unpack_proto(input_proto)

        if self.height is None or self.width is None:
            self.height = input_image.shape[0]
            self.width = input_image.shape[1]
        
        inference_start = time.monotonic_ns()
        tracking_output_array = np.array([])
        det_array,bbox,confidence,feats,class_ids = self._prepare_detection_input(sae_msg)
        
        with MODEL_DURATION.time():
            if self.config.tracker_algorithm == TrackingAlgorithm.DEEPOCSORT:
                tracking_output_array = self.tracker.update(det_array, input_image,sae_msg)
            elif self.config.tracker_algorithm == TrackingAlgorithm.DEEPSORT:
                self.tracker.update(bbox, confidence, feats,class_ids,stream_id)
                tracking_output_array, out_features = self._trackingreusltprocess()
            elif self.config.tracker_algorithm == TrackingAlgorithm.SMILETRACK:
                self.tracker.update(bbox,confidence,feats,class_ids,input_image,det_array,frame_id=sae_msg.frame.frame_id)
                tracking_output_array, out_features = self._trackingreusltprocess1()

        if self.config.tracker_config.multi_camera_tracking:
            sae_msg = self._tracklet_info_update(stream_id,tracking_output_array,out_features,sae_msg)

        OBJECT_COUNTER.inc(len(tracking_output_array))
        
        inference_time_us = (time.monotonic_ns() - inference_start) // 1000
        return self._create_output(tracking_output_array, sae_msg, inference_time_us)
        
    def _setup(self):
        conf = self.config.tracker_config
        if self.config.tracker_algorithm == TrackingAlgorithm.DEEPOCSORT:
            self.tracker = DeepOCSORT(
                model_weights=Path(self.config.tracker_config.model_weights),
                device=conf.device,
                fp16=conf.fp16,
                per_class=conf.per_class,
                det_thresh=conf.det_thresh,
                max_age=conf.max_age,
                min_hits=conf.min_hits,
                iou_threshold=conf.iou_threshold,
                delta_t=conf.delta_t,
                asso_func=conf.asso_func,
                inertia=conf.inertia,
                w_association_emb=conf.w_association_emb,
                alpha_fixed_emb=conf.alpha_fixed_emb,
                aw_param=conf.aw_param,
                embedding_off=conf.embedding_off,
                cmc_off=conf.cmc_off,
                aw_off=conf.aw_off,
                new_kf_off=conf.new_kf_off
            )
        elif self.config.tracker_algorithm == TrackingAlgorithm.OCSORT:
            self.tracker = OCSORT(
                det_thresh=conf.det_thresh,
                max_age=conf.max_age,
                min_hits=conf.min_hits,
                asso_threshold=conf.asso_threshold,
                delta_t=conf.delta_t,
                asso_func=conf.asso_func,
                inertia=conf.inertia,
                use_byte=conf.use_byte
            )
        elif self.config.tracker_algorithm == TrackingAlgorithm.DEEPSORT:
            self.tracker = objecttracker.Modified_Deepsort.Modified_Tracker(max_cosine_distance = conf.max_cosine_distance,
                                                              min_confidence = conf.min_confidence,
                                                              max_iou_distance = conf.max_iou_distance,
                                                              max_age = conf.max_age,
                                                              n_init= conf.n_init)
        elif self.config.tracker_algorithm == TrackingAlgorithm.SMILETRACK: 
            self.tracker = objecttracker.Modified_SMILEtrack.Modified_Tracker(  track_low_thresh = conf.track_low_thresh,
                                                                                track_high_thresh = conf.track_high_thresh,
                                                                                new_track_thresh = conf.new_track_thresh,
                                                                                track_buffer = conf.track_buffer,
                                                                                proximity_thresh = conf.proximity_thresh,
                                                                                appearance_thresh = conf.appearance_thresh,
                                                                                with_reid = conf.with_reid,
                                                                                match_thresh = conf.match_thresh,
                                                                                frame_rate = conf.frame_rate)
        else:
            logger.error(f'Unknown tracker algorithm: {self.config.tracker_algorithm}')
            exit(1)

    @PROTO_DESERIALIZATION_DURATION.time()
    def _unpack_proto(self, sae_message_bytes):
        sae_msg = SaeMessage()
        sae_msg.ParseFromString(sae_message_bytes)

        input_frame = sae_msg.frame
        input_image = get_raw_frame_data(input_frame)

        return input_image, sae_msg
    
    def _prepare_detection_input(self,sae_msg: SaeMessage):
        '''
        This function serves SaeMessage as input
        returns det_arrray (contains detection information with sequence!)
                bbox,confidence,feats,class_id

        separately
        '''
        bbox = []
        confidence = []
        feats = []
        class_ids = []
        det_array = np.zeros((len(sae_msg.detections), 8))
        for idx, detection in enumerate(sae_msg.detections):
            '''
            min_x = detection.bounding_box.min_x * image_shape[1]
            max_x = detection.bounding_box.max_x * image_shape[1]
            min_y = detection.bounding_box.min_y * image_shape[0]
            max_y = detection.bounding_box.max_y * image_shape[0]
            '''
            min_x = detection.bounding_box.min_x * self.width
            max_x = detection.bounding_box.max_x * self.width
            min_y = detection.bounding_box.min_y * self.height
            max_y = detection.bounding_box.max_y * self.height           
            w = max_x - min_x
            h = max_y - min_y
            bbox.append((min_x, min_y, w, h))
            confidence.append(detection.confidence)
            # logger.info(type(detection.feature))
            feats.append(detection.feature)
            class_ids.append(detection.class_id)

            det_array[idx, 0] = detection.bounding_box.min_x * self.width
            det_array[idx, 1] = detection.bounding_box.min_y * self.height
            det_array[idx, 2] = detection.bounding_box.max_x * self.width
            det_array[idx, 3] = detection.bounding_box.max_y * self.height
            det_array[idx, 4] = detection.confidence
            det_array[idx, 5] = detection.class_id
            det_array[idx, 6] = detection.geo_coordinate.latitude
            det_array[idx, 7] = detection.geo_coordinate.longitude
        return det_array,bbox,confidence,feats,class_ids
    
    # def _trackingreusltprocess(self):
    #     tracking_output_array = np.zeros((len(self.tracker.tracks), 8))
    #     features = []
    #     for index, track in enumerate(self.tracker.tracks):
    #         x, y, w, h = track.bbox
    #         x1, y1, x2, y2 = x, y, x + w, y + h
    #         x1, y1 = max(x1, 0), max(y1, 0)
    #         track_id = track.track_id
    #         features.append(track.feat)
    #         confidence = track.confidence
    #         class_id = track.class_id
    #         age = track.age
    #         tracking_output_array[index] = np.array([x1, y1, x2, y2, track_id, confidence,class_id, age])
        
    #     return tracking_output_array,features
    
    def _trackingreusltprocess1(self):
        tracking_output_array = np.zeros((len(self.tracker.tracks), 10))
        features = []
        for index, track in enumerate(self.tracker.tracks):
            x, y, w, h = track.bbox
            x1, y1, x2, y2 = x, y, x + w, y + h
            x1, y1 = max(x1, 0), max(y1, 0)
            track_id = track.track_id
            features.append(track.feat)
            confidence = track.confidence
            class_id = track.class_id
            lat = track.lat
            lon = track.lon
            age = 0
            tracking_output_array[index] = np.array([x1, y1, x2, y2, track_id, confidence,class_id, age, lat, lon])
            
        return tracking_output_array,features

    def _tracklet_info_update(self, stream_id, tracking_output_array, out_features, sae_msg: SaeMessage):
        """
        Updates the tracklet information in the SaeMessage.trajectory for the current frame.

        Args:
            stream_id (str): The ID of the camera stream.
            tracking_output_array (np.ndarray): Array containing tracking outputs.
            out_features (list): List of feature vectors for each tracked object.
            input_image (np.ndarray): The input image.
            sae_msg (SaeMessage): The SaeMessage object to update.

        Returns:
            SaeMessage: Updated SaeMessage with tracklet information.
        """
        save_path = self.config.save_config.save_path if self.config.save_config.save else None

        # Initialize the camera's tracklets in the trajectory
        sae_msg.trajectory.cameras[stream_id].CopyFrom(TrackletsByCamera())

        if save_path:
            with open(save_path, "a") as f:
                for index, output_array in enumerate(tracking_output_array):
                    x1, y1, x2, y2, track_id, confidence, class_id, age, lat, lon = output_array
                    feature = out_features[index]
                    track_id_str = str(track_id)

                    # Create and populate a new tracklet
                    tracklet = Tracklet()
                    tracklet.mean_feature.extend(feature)
                    tracklet.status = "Active"
                    tracklet.start_time = sae_msg.frame.timestamp_utc_ms
                    tracklet.end_time = sae_msg.frame.timestamp_utc_ms

                    # Add detection information to the tracklet
                    detection = tracklet.detections_info.add()
                    detection.bounding_box.min_x = float(x1) / self.width
                    detection.bounding_box.min_y = float(y1) / self.height
                    detection.bounding_box.max_x = float(x2) / self.width
                    detection.bounding_box.max_y = float(y2) / self.height
                    detection.confidence = float(confidence)
                    detection.class_id = int(class_id)
                    detection.feature.extend(feature)
                    detection.geo_coordinate.latitude = float(lat)
                    detection.geo_coordinate.longitude = float(lon)
                    detection.timestamp_utc_ms = sae_msg.frame.timestamp_utc_ms
                    detection.frame_id = sae_msg.frame.frame_id

                    # Save the tracklet to the camera's tracklets
                    sae_msg.trajectory.cameras[stream_id].tracklets[track_id_str].CopyFrom(tracklet)

                    # Write tracking information to the save file
                    line = f"{sae_msg.frame.frame_id} {track_id_str} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} {class_id}"
                    f.write(line + "\n")
        else:
            for index, output_array in enumerate(tracking_output_array):
                x1, y1, x2, y2, track_id, confidence, class_id, age, lat, lon = output_array
                feature = out_features[index]
                track_id_str = str(track_id)

                # Create and populate a new tracklet
                tracklet = Tracklet()
                tracklet.mean_feature.extend(feature)
                tracklet.status = "Active"
                tracklet.start_time = sae_msg.frame.timestamp_utc_ms
                tracklet.end_time = sae_msg.frame.timestamp_utc_ms

                # Add detection information to the tracklet
                detection = tracklet.detections_info.add()
                detection.bounding_box.min_x = float(x1) / self.width
                detection.bounding_box.min_y = float(y1) / self.height
                detection.bounding_box.max_x = float(x2) / self.width
                detection.bounding_box.max_y = float(y2) / self.height
                detection.confidence = float(confidence)
                detection.class_id = int(class_id)
                detection.feature.extend(feature)
                detection.geo_coordinate.latitude = float(lat)
                detection.geo_coordinate.longitude = float(lon)
                detection.timestamp_utc_ms = sae_msg.frame.timestamp_utc_ms
                detection.frame_id = sae_msg.frame.frame_id

                # Save the tracklet to the camera's tracklets
                sae_msg.trajectory.cameras[stream_id].tracklets[track_id_str].CopyFrom(tracklet)

        return sae_msg
    @PROTO_SERIALIZATION_DURATION.time()
    def _create_output(self, tracking_output, input_sae_msg: SaeMessage, inference_time_us) -> bytes:
        output_sae_msg = SaeMessage()
        output_sae_msg.frame.CopyFrom(input_sae_msg.frame)
        output_sae_msg.trajectory.CopyFrom(input_sae_msg.trajectory)

        # The length of detections and tracking_output can be different 
        # (as the latter only includes objects that could be matched to an id)
        # Therefore, we can only reuse the VideoFrame and have to recreate everything else
        for pred in tracking_output:
            detection = output_sae_msg.detections.add()
            detection.bounding_box.min_x = float(pred[0]) / self.width
            detection.bounding_box.min_y = float(pred[1]) / self.height
            detection.bounding_box.max_x = float(pred[2]) / self.width
            detection.bounding_box.max_y = float(pred[3]) / self.height

            # detection.object_id = uuid.uuid3(self.object_id_seed, str(int(pred[4]))).bytes
            detection.object_id = int(pred[4])

            detection.confidence = float(pred[5])
            detection.class_id = int(pred[6])
            detection.geo_coordinate.latitude = float(pred[8])
            detection.geo_coordinate.longitude = float(pred[9])
            detection.timestamp_utc_ms = input_sae_msg.frame.timestamp_utc_ms
            detection.frame_id = input_sae_msg.frame.frame_id

        output_sae_msg.metrics.CopyFrom(input_sae_msg.metrics)
        output_sae_msg.metrics.tracking_inference_time_us = inference_time_us
        
        return output_sae_msg.SerializeToString()