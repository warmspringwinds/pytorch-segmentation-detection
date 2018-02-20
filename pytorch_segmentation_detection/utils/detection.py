import torch


def convert_bbox_xywh_tensor_to_xyxy(bbox_xywh_tensor):
    
    bbox_xyxy_tensor = bbox_xywh_tensor.clone()
    
    bbox_xyxy_tensor[:, 2] = bbox_xyxy_tensor[:, 0] + bbox_xyxy_tensor[:, 2]
    bbox_xyxy_tensor[:, 3] = bbox_xyxy_tensor[:, 1] + bbox_xyxy_tensor[:, 3]
    
    return bbox_xyxy_tensor