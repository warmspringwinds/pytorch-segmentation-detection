import torch
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.patches as patches


def compute_network_output_feature_map_size(input_img_size, stride):
    """Function to compute the size of the output feature map of the network.
    
    Given an image size and stride of a network, computes the output feature map size.
    Basically just coputes input_img_size / stride values.
    
    Parameters
    ----------
    input_img_size : tuple of ints
        Tuple with height and width sizes of the image
    
    stride : int
        Output stride of the network
        
    Returns
    -------
    feature_map_size : tuple of ints
        Size of the output feature map.
    """
    
    
    input_size = np.asarray(input_img_size).astype(np.float)

    feature_map_size = input_size / stride
    
    return np.floor(feature_map_size).astype(np.int)


# Abbreviations:

# center xywh -- center x/y coordinates or a rectangle with width and height
# Used everywhere during training because the method computes errors based on the
# difference of center coordinates of the groundtruth boxes and bounding boxes.

# topleft xywh -- top left coordinates of a bounding box and its width and height.
# This format is used in the coco-like annotations for pascal voc. Also used
# for easier visualization.

# xyxy -- topleft x/y coordinates and bottom right x/y coordinates of a rectangle.
# This format is used for computation of intersection over union metric.

# Short desciption of the methods:

# -- convert_bbox_topleft_xywh_tensor_to_center_xywh() is used to convert records
# that we recieve from coco-like dataloader to our canonical xy center representation
# that is required for training of our model

# -- convert_bbox_center_xywh_tensor_to_xyxy() is used to convert our bounding boxes
# in canonical center xywh representation to xyxy one in order to easily compute
# intersection over using using compute_bboxes_ious() function


def convert_bbox_topleft_xywh_tensor_to_center_xywh(bbox_topleft_xywh_tensor):
    """Function to convert bounding boxes in format (x_topleft, y_topleft, width, height)
    to a format of (x_center, y_center, width, height).
    
    Works with a tensors of a (N, 4) shape.
    
    Parameters
    ----------
    bbox_xywh_tensor : FloatTensor of shape (N, 4)
        Tensor with bounding boxes in topleft_xywh format
        
    Returns
    -------
    bbox_xyxy_tensor : FloatTensor of shape (N, 4)
        Tensor with bounding boxes in center_xywh format
    """
    
    bbox_center_xywh_tensor = bbox_topleft_xywh_tensor.clone()
    
    bbox_center_xywh_tensor[:, 0] = bbox_topleft_xywh_tensor[:, 0] + bbox_topleft_xywh_tensor[:, 2] * 0.5
    bbox_center_xywh_tensor[:, 1] = bbox_topleft_xywh_tensor[:, 1] + bbox_topleft_xywh_tensor[:, 3] * 0.5
    
    return bbox_center_xywh_tensor

def convert_bbox_center_xywh_tensor_to_xyxy(bbox_center_xywh_tensor):
    """Function to convert bounding boxes in format (x_center, y_center, width, height)
    to a format of (x_min, y_min, x_max, y_max).
    
    Works with a tensors of a (N, 4) shape.
    
    Parameters
    ----------
    bbox_xywh_tensor : FloatTensor of shape (N, 4)
        Tensor with bounding boxes in center_xywh format
        
    Returns
    -------
    bbox_xyxy_tensor : FloatTensor of shape (N, 4)
        Tensor with bounding boxes in xyxy format
    """
    
    bbox_xyxy_tensor = bbox_center_xywh_tensor.clone()
    
    # Getting top left corner
    bbox_xyxy_tensor[:, 0] = bbox_center_xywh_tensor[:, 0] - bbox_center_xywh_tensor[:, 2] * 0.5
    bbox_xyxy_tensor[:, 1] = bbox_center_xywh_tensor[:, 1] - bbox_center_xywh_tensor[:, 3] * 0.5
    
    # Getting bottom right corner
    bbox_xyxy_tensor[:, 2] = bbox_center_xywh_tensor[:, 0] + bbox_center_xywh_tensor[:, 2] * 0.5
    bbox_xyxy_tensor[:, 3] = bbox_center_xywh_tensor[:, 1] + bbox_center_xywh_tensor[:, 3] * 0.5
    
    return bbox_xyxy_tensor


def convert_bbox_topleft_xywh_tensor_to_xyxy(bbox_topleft_xywh_tensor):
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
    
    bbox_xyxy_tensor = bbox_topleft_xywh_tensor.clone()
    
    bbox_xyxy_tensor[:, 2] = bbox_topleft_xywh_tensor[:, 0] + bbox_topleft_xywh_tensor[:, 2]
    bbox_xyxy_tensor[:, 3] = bbox_topleft_xywh_tensor[:, 1] + bbox_topleft_xywh_tensor[:, 3]
    
    return bbox_xyxy_tensor


def convert_bbox_center_xywh_tensor_to_topleft_xywh(bbox_center_xywh_tensor):
    """Function to convert bounding boxes in format (x_center, y_center, width, height)
    to a format of (x_topleft, y_topleft, width, height).
    
    Works with a tensors of a (N, 4) shape.
    
    Parameters
    ----------
    bbox_center_xywh_tensor : FloatTensor of shape (N, 4)
        Tensor with bounding boxes in center_xywh format
        
    Returns
    -------
    bbox_topleft_xywh_tensor : FloatTensor of shape (N, 4)
        Tensor with bounding boxes in topleft_xywh format
    """
    
    bbox_topleft_xywh_tensor = bbox_center_xywh_tensor.clone()
    
    bbox_topleft_xywh_tensor[:, 0] = bbox_center_xywh_tensor[:, 0] - bbox_center_xywh_tensor[:, 2] * 0.5
    bbox_topleft_xywh_tensor[:, 1] = bbox_center_xywh_tensor[:, 1] - bbox_center_xywh_tensor[:, 3] * 0.5
    
    return bbox_topleft_xywh_tensor


def display_bboxes_center_xywh(img, bboxes_center_xywh):
    """Function for displaying bounding boxes on the given image.
    
    Displays the bounding boxes in the format of center_xywh on the given image.
    
    Parameters
    ----------
    bboxes_center_xywh : FloatTensor of shape (N, 4)
        Tensor with bounding boxes in center_xywh format
    """
    
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(9, 6.9))
    
    # Display the image
    ax.imshow(img)
    
    bboxes_topleft_xywh = convert_bbox_center_xywh_tensor_to_topleft_xywh(bboxes_center_xywh)
        
    for bbox_topleft_xywh in bboxes_topleft_xywh:
    
        # Create a Rectangle patch
        rect = patches.Rectangle(bbox_topleft_xywh[:2],
                                 bbox_topleft_xywh[2],
                                 bbox_topleft_xywh[3],
                                 linewidth=2,
                                 edgecolor='b',
                                 facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
    
    plt.show()

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