import logging
import objecttracker.Modified_Deepsort
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch
from boxmot import OCSORT, DeepOCSORT
from prometheus_client import Counter, Histogram, Summary
from objecttracker.vision_api.python.visionapi.visionapi.messages_pb2 import SaeMessage
from visionlib.pipeline.tools import get_raw_frame_data

from .config import ObjectTrackerConfig, TrackingAlgorithm

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
        
    def __call__(self, input_proto, *args, **kwargs) -> Any:
        return self.get(input_proto)

    @GET_DURATION.time()
    @torch.no_grad()
    def get(self, input_proto):        

        input_image, sae_msg = self._unpack_proto(input_proto)
        
        inference_start = time.monotonic_ns()
        tracking_output_array = np.array([])
        
        det_array,bbox,confidence,feats,class_ids = self._prepare_detection_input(sae_msg,input_image.shape[:2])
        with MODEL_DURATION.time():
            if self.config.tracker_algorithm != TrackingAlgorithm.DEEPSORT:
                tracking_output_array = self.tracker.update(det_array, input_image)
            elif self.config.tracker_algorithm == TrackingAlgorithm.DEEPSORT:
                self.tracker.update(bbox, confidence, feats,class_ids)
                tracking_output_array = np.zeros((len(self.tracker.tracks), 7))
                for index, track in enumerate(self.tracker.tracks):
                    x, y, w, h = track.bbox
                    x1, y1, x2, y2 = x, y, x + w, y + h
                    x1, y1 = max(x1, 0), max(y1, 0)
                    track_id = track.track_id
                    feat = track.feat
                    confidence = track.confidence
                    class_id = track.class_id
                    tracking_output_array[index] = np.array([x1, y1, x2, y2, track_id, confidence,class_id])

            logger.info(tracking_output_array)
                # logger.info(len(feat))

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
    
    def _prepare_detection_input(self,sae_msg: SaeMessage,image_shape):
        width = image_shape[1]
        heigh = image_shape[0]
        bbox = []
        confidence = []
        feats = []
        class_ids = []
        det_array = np.zeros((len(sae_msg.detections), 6))
        for idx, detection in enumerate(sae_msg.detections):
            '''
            min_x = detection.bounding_box.min_x * image_shape[1]
            max_x = detection.bounding_box.max_x * image_shape[1]
            min_y = detection.bounding_box.min_y * image_shape[0]
            max_y = detection.bounding_box.max_y * image_shape[0]
            '''
            min_x = detection.bounding_box.min_x 
            max_x = detection.bounding_box.max_x 
            min_y = detection.bounding_box.min_y 
            max_y = detection.bounding_box.max_y            
            w = max_x - min_x
            h = max_y - min_y
            bbox.append((min_x, min_y, w, h))
            confidence.append(detection.confidence)
            # logger.info(type(detection.feature))
            feats.append(detection.feature)
            class_ids.append(detection.class_id)

            det_array[idx, 0] = detection.bounding_box.min_x
            det_array[idx, 1] = detection.bounding_box.min_y
            det_array[idx, 2] = detection.bounding_box.max_x
            det_array[idx, 3] = detection.bounding_box.max_y
            det_array[idx, 4] = detection.confidence
            det_array[idx, 5] = detection.class_id
            # logger.info(detection)

            # det_array[idx, 6] = detection.feature
            # logger.info(f'feature extract with shape{len(detection.feature)}')
            # logger.info(detection.feature)
        return det_array,bbox,confidence,feats,class_ids
    
    @PROTO_SERIALIZATION_DURATION.time()
    def _create_output(self, tracking_output, input_sae_msg: SaeMessage, inference_time_us):
        output_sae_msg = SaeMessage()
        output_sae_msg.frame.CopyFrom(input_sae_msg.frame)

        # The length of detections and tracking_output can be different 
        # (as the latter only includes objects that could be matched to an id)
        # Therefore, we can only reuse the VideoFrame and have to recreate everything else
        
        for pred in tracking_output:
            detection = output_sae_msg.detections.add()
            detection.bounding_box.min_x = float(pred[0])
            detection.bounding_box.min_y = float(pred[1])
            detection.bounding_box.max_x = float(pred[2])
            detection.bounding_box.max_y = float(pred[3])

            detection.object_id = uuid.uuid3(self.object_id_seed, str(int(pred[4]))).bytes

            detection.confidence = float(pred[5])
            detection.class_id = int(pred[6])

        output_sae_msg.metrics.CopyFrom(input_sae_msg.metrics)
        output_sae_msg.metrics.tracking_inference_time_us = inference_time_us
        
        return output_sae_msg.SerializeToString()
