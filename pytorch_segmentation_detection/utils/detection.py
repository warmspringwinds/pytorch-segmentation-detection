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


def compute_bboxes_ious(bboxes_xyxy_group_1, bboxes_xyxy_group_2):
    """Function to compute the intersection over union (IOU) metric
    for each pair of rectangles from bboxes_group_1 and bboxes_group_2
    
    Parameters
    ----------
    bboxes_xyxy_group_1 : FloatTensor of shape (N, 4)
        Tensor with bounding boxes in xyxy format
        
    bboxes_xyxy_group_2 : FloatTensor of shape (M, 4)
        Tensor with bounding boxes in xyxy format
        
    Returns
    -------
    ious : FloatTensor of shape (N, M)
        Tensor with iou metric computed between all possible pairs
        from group 1 and 2
    """
    
    # Computing the bboxes of the intersections between
    # each pair of boxes from group 1 and 2
    
    # top_left: (N, M, 2)
    top_left = torch.max(bboxes_xyxy_group_1[:, None , :2],
                         bboxes_xyxy_group_2[:, :2])
    
    # bottom_right: (N, M, 2)
    bottom_right = torch.min(bboxes_xyxy_group_1[:, None, 2:],
                             bboxes_xyxy_group_2[:, 2:])
    
    intersections_bboxes_width_height = torch.clamp( bottom_right - top_left, min=0)
    
    # intersections_bboxes_areas: (N, M)
    intersections_bboxes_areas = intersections_bboxes_width_height[:, :, 0] * intersections_bboxes_width_height[:, :, 1]
    
    # bboxes_group_1_areas: (N,)
    bboxes_group_1_areas = (bboxes_xyxy_group_1[:,2]-bboxes_xyxy_group_1[:,0]) * (bboxes_xyxy_group_1[:,3]-bboxes_xyxy_group_1[:,1])
    
    # bboses_group_2_areas: (M,)
    bboxes_group_2_areas = (bboxes_xyxy_group_2[:,2]-bboxes_xyxy_group_2[:,0]) * (bboxes_xyxy_group_2[:,3]-bboxes_xyxy_group_2[:,1])
    
    # ious: (N, M)
    ious = intersections_bboxes_areas / (bboxes_group_1_areas[:, None] + bboxes_group_2_areas - intersections_bboxes_areas)
    
    return ious