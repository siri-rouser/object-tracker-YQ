from objecttracker.vision_api.python.visionapi.visionapi.messages_pb2 import SaeMessage, TrackletsByCamera,Trajectory,Tracklet
from visionlib.pipeline.tools import get_raw_frame_data


def tracklet_info_update(stream_id,tracking_output_array,out_features,out_age,sae_msg: SaeMessage):
    if stream_id not in sae_msg.trajectory.cameras:
        sae_msg.trajectory.cameras[stream_id].CopyFrom(TrackletsByCamera())

    # tracking_output_arrary = [x1, y1, x2, y2, track_id, confidence,class_id]
    tracklet = Tracklet()

    for index,output_array in enumerate(tracking_output_array):
        x1, y1, x2, y2, track_id, confidence, class_id = output_array
        feature = out_features[index]
        track_id = str(track_id)
        if track_id not in sae_msg.trajectory.cameras[stream_id].tracklets:
            # Create a new Tracklet if it doesn't exist
            tracklet = Tracklet()
            tracklet.mean_feature.extend(out_features[index])
            tracklet.states = 'Active'
            tracklet.start_time = sae_msg.frame.timestamp_utc_ms
            tracklet.age = out_age[index]
            # for detection_info
            detection = tracklet.detections_info.add()
            detection.bounding_box.min_x = x1
            detection.bounding_box.min_y = y1
            detection.bounding_box.max_x = x2
            detection.bounding_box.max_y = y2
            detection.confidence = confidence
            detection.class_id = int(class_id)
            detection.feature.extend(out_features[index])

            # Add the new tracklet to the tracklets map
            sae_msg.trajectory.cameras[stream_id].tracklets[track_id].CopyFrom(tracklet)
        else:
            previous_feature = sae_msg.trajectory.cameras[stream_id].tracklets[track_id].mean_feature
            number_det = len(sae_msg.trajectory.cameras[stream_id].tracklets[track_id].detections_info)
            sae_msg.trajectory.cameras[stream_id].tracklets[track_id].mean_feature = (number_det-1)/number_det*previous_feature + (1/number_det)*out_features
            
            # for detection_info
            detection = sae_msg.trajectory.cameras[stream_id].tracklets[track_id].detections_info.add()
            detection.bounding_box.min_x = x1
            detection.bounding_box.min_y = y1
            detection.bounding_box.max_x = x2
            detection.bounding_box.max_y = y2
            detection.confidence = confidence
            detection.class_id = int(class_id)
            detection.feature.extend(out_features[index])
            sae_msg.trajectory.cameras[stream_id].tracklets[track_id].age = out_age[index]
            sae_msg.trajectory.cameras[stream_id].tracklets[track_id].end_time = sae_msg.frame.timestamp_utc_ms
        
        # tracklet[track_id].detections_info.bounding_box.min_x = x1
    return sae_msg


def tracklet_status_update(stream_id,sae_msg: SaeMessage):
    time_now = sae_msg.frame.timestamp_utc_ms
    for track_id in sae_msg.trajectory.cameras[stream_id].tracklets:
        if (sae_msg.trajectory.cameras[stream_id].tracklets[track_id].age > 30) and (time_now - sae_msg.trajectory.cameras[stream_id].tracklets[track_id].end_time < 60000):
            #60000 is just a assuming time in here, this value is get from camera link model
            sae_msg.trajectory.cameras[stream_id].tracklets[track_id].states = 'inactive' # Lost1 means temporal lost
        elif (time_now - sae_msg.trajectory.cameras[stream_id].tracklets[track_id].end_time > 60000) and (time_now - sae_msg.trajectory.cameras[stream_id].tracklets[track_id].end_time < 180000):
            sae_msg.trajectory.cameras[stream_id].tracklets[track_id].states = 'Searching'
        elif time_now - sae_msg.trajectory.cameras[stream_id].tracklets[track_id].end_time > 180000:
            sae_msg.trajectory.cameras[stream_id].tracklets[track_id].states = 'Lost'

    return sae_msg


def tracklet_match(proto_data):
    input_image, sae_msg = unpack_proto(proto_data)

    tracklets1 = sae_msg.trajectory.cameras['stream1'].tracklets
    # Iterate over track_id keys and print them
    for track_id in tracklets1.keys():
        print(f"Track ID: {track_id}")

    return 


def unpack_proto(sae_message_bytes):
    sae_msg = SaeMessage()
    sae_msg.ParseFromString(sae_message_bytes)

    input_frame = sae_msg.frame
    input_image = get_raw_frame_data(input_frame)

    return input_image, sae_msg