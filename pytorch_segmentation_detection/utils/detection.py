import torch


def convert_bbox_xywh_tensor_to_xyxy(bbox_xywh_tensor):
    """Function to convert bounding boxes in format (x_topleft, y_topleft, width, height)
    to a format of (x_min, y_min, x_max, y_max).
    
    Works with a tensors of a (N, 4) shape.
    
    Parameters
    ----------
    bbox_xywh_tensor : FloatTensor of shape (N, 4)
        Tensor with bounding boxes in xywh format
        
    Returns
    -------
    bbox_xyxy_tensor : FloatTensor of shape (N, 4)
        Tensor with bounding boxes in xyxy format
    """
    
    bbox_xyxy_tensor = bbox_xywh_tensor.clone()
    
    bbox_xyxy_tensor[:, 2] = bbox_xyxy_tensor[:, 0] + bbox_xyxy_tensor[:, 2]
    bbox_xyxy_tensor[:, 3] = bbox_xyxy_tensor[:, 1] + bbox_xyxy_tensor[:, 3]
    
    return bbox_xyxy_tensor