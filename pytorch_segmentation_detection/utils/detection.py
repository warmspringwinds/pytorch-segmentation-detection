import torch
import math
import random
import numpy as np

from matplotlib import pyplot as plt
import matplotlib.patches as patches

from PIL import Image, ImageOps
from torch.autograd import Variable


def random_crop_with_bounding_boxes(input_img, crop_size, bboxes_center_xywh, fill_label=0):
    """A function that pads image to the size with fill_label if the input image is smaller
    and updates the coordinates of bounding boxes in a format of center_xywh that were defined
    on the original image.
        
    Parameters
    ----------
    input_img : PIL image
        Tuple with height and width sizes of the image
    size : tuple of ints
        Tuple of ints representing width and height of a desired padded image
    bboxes_center_xywh: torch.FloatTensor of size (N, 4)
        Tensor containing bounding boxes defined in center_xywh format
    fill_label : int
        A value used to fill image in the padded areas.
        
    Returns
    -------
    processed_img: PIL image
        Image that was padded to the desired size
        
    bboxes_center_xywhh_padded : torch.FloatTensor
        Tensor that contains updated with respect to performed padding
        coordinates of bounding boxes in a center_xywh format.
    """
    
    padded_img_pil, padded_bboxes_center_xywh = pad_to_size_with_bounding_boxes(input_img, crop_size, bboxes_center_xywh)
        
    # # We assume that inputs were of the same size before padding.
    # # So they are of the same size after the padding
    w, h = padded_img_pil.size

    th, tw = crop_size

    if w == tw and h == th:
        return padded_img_pil, padded_bboxes_center_xywh

    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)

    res_img_pil = padded_img_pil.crop((x1, y1, x1 + tw, y1 + th))

    padded_bboxes_center_xyxy = convert_bbox_center_xywh_tensor_to_xyxy(padded_bboxes_center_xywh)

    padded_bboxes_center_xyxy_cropped = padded_bboxes_center_xyxy - torch.Tensor([x1,y1,x1,y1])
    #padded_bboxes_center_xyxy_cropped[:,0::2].clamp_(min=0, max=tw-1)
    #padded_bboxes_center_xyxy_cropped[:,1::2].clamp_(min=0, max=th-1)

    padded_bboxes_center_xywh_cropped = convert_bbox_xyxy_tensor_to_center_xywh(padded_bboxes_center_xyxy_cropped)
    
    return res_img_pil, padded_bboxes_center_xywh_cropped


def convert_bbox_xyxy_tensor_to_center_xywh(bbox_xyxy_tensor):
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
    
    bbox_center_xywh_tensor = bbox_xyxy_tensor.clone()
    
    # Getting top left corner
    bbox_center_xywh_tensor[:, 0] = (bbox_xyxy_tensor[:, 0] + bbox_xyxy_tensor[:, 2]) * 0.5
    bbox_center_xywh_tensor[:, 1] = (bbox_xyxy_tensor[:, 1] + bbox_xyxy_tensor[:, 3]) * 0.5
    
    # Getting bottom right corner
    bbox_center_xywh_tensor[:, 2] = bbox_xyxy_tensor[:, 2] - bbox_xyxy_tensor[:, 0]
    bbox_center_xywh_tensor[:, 3] = bbox_xyxy_tensor[:, 3] - bbox_xyxy_tensor[:, 1]
    
    return bbox_center_xywh_tensor

def box_nms(bboxes, scores, threshold=0.5, mode='union'):
    """Function that performes non-maximum suppression of predicted
    bounding boxes by prunning predictions that significantly overlap.

    Credit:
    https://github.com/kuangliu/pytorch-retinanet

    Reference:
    https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py

    Parameters
    ----------
    bboxes : torch.FloatTensor of size (#boxes, 4)
     Tensor containing bounding boxes in a xyxy format

    scores:  torch.FloatTensor of size (#boxes,)
     Tensor containing confidence scores for each of the bounding
     boxes (probabilities).

    Returns
    -------
    boxes : torch.LongTensor of size (#selected_boxes,)
     Indexes of boxes that were not prunned
    """
        
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]

    areas = (x2-x1) * (y2-y1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w*h

        if mode == 'union':
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == 'min':
            ovr = inter / areas[order[1:]].clamp(max=areas[i])
        else:
            raise TypeError('Unknown nms mode: %s.' % mode)

        ids = (ovr<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]

    return torch.LongTensor(keep)


def pad_to_size_with_bounding_boxes(input_img, size, bboxes_center_xywh, fill_label=0):
    """A function that pads image to the size with fill_label if the input image is smaller
    and updates the coordinates of bounding boxes in a format of center_xywh that were defined
    on the original image.
        
    Parameters
    ----------
    input_img : PIL image
        Tuple with height and width sizes of the image

    size : tuple of ints
        Tuple of ints representing width and height of a desired padded image

    bboxes_center_xywh: torch.FloatTensor of size (N, 4)
        Tensor containing bounding boxes defined in center_xywh format

    fill_label : int
        A value used to fill image in the padded areas.
        
    Returns
    -------
    processed_img: PIL image
        Image that was padded to the desired size
        
    bboxes_center_xywhh_padded : torch.FloatTensor
        Tensor that contains updated with respect to performed padding
        coordinates of bounding boxes in a center_xywh format.
    """
    
    input_size = np.asarray(input_img.size)
    padded_size = np.asarray(size)

    difference = padded_size - input_size

    parts_to_expand = difference > 0

    expand_difference = difference * parts_to_expand

    expand_difference_top_and_left = expand_difference // 2

    expand_difference_bottom_and_right = expand_difference - expand_difference_top_and_left
    
    # Form the PIL config vector
    pil_expand_array = np.concatenate( (expand_difference_top_and_left,
                                        expand_difference_bottom_and_right) )
    
    processed_img = input_img
    
    # Check if we actually need to expand our image.
    if pil_expand_array.any():
        
        pil_expand_tuple = tuple(pil_expand_array)
        
        processed_img = ImageOps.expand(input_img, border=pil_expand_tuple, fill=fill_label)
    
    
    # TODO: obviously there is something wrong with the name of the top_and_left
    # variables as they are not representing top and left difference -- top and left
    # should be probably swapped -- needs more inspecting. The function was checked to 
    # work correctly though.
    
    bboxes_center_xywhh_padded = bboxes_center_xywh.clone()
    bboxes_center_xywhh_padded[:, 0] = bboxes_center_xywh[:, 0] + expand_difference_top_and_left[0]
    bboxes_center_xywhh_padded[:, 1] = bboxes_center_xywh[:, 1] + expand_difference_top_and_left[1]
    
    
    return processed_img, bboxes_center_xywhh_padded


class AnchorBoxesManager():
    """
    AnchorBoxesManager class keeps track of all anchor boxes.
    
    First, it generates boxes for each element of output feature map of the network,
    given its stride. Each element of the ouput feature map is associated
    with a region of the input image that is being subsampled. The generated
    bounding boxes are centered at each of these regions.
    
    The class accepts ```anchor_areas```, ```aspect_ratios``` parameters.
    All possible pairs of these combinations are generated and result in
    ```size(anchor_areas) * size(aspect_ratios) ``` anchor boxes for each
    element of resulted feature map.
    
    Given the ground truth bounding boxes and respective image, the class is
    able to generate values for each box that point to how the box should be
    moved in order to be aligned with the closest ground truth bounding box.
    
    These values are used as target values during training.
    """
    
    def __init__(self,
                 input_image_size=(600, 600),
                 anchor_areas=[128*128, 256*256, 512*512],
                 aspect_ratios=[1/2., 1/1., 2/1.],
                 stride=32
                ):
        """Constructor function for the anchor box manager class.
        
        Given the input image size, output stride of the network, anchor areas
        and their aspect ratios precomputes the coordinates and sizes of all anchor
        boxes.
        
        Later on, given groundtruth bounding boxes, it computes parametrized
        in a certain way delta values that indicate how each anchor box should
        be shifted in order to reach the closest groundtruth bounding box. These
        values are used as target values during training.
        
        Parameters
        ----------
        input_img_size : tuple of ints
            Tuple with height and width sizes of the image
            
        anchor_areas : tuple of floats
            Array of floats indicating the anchor areas sizes.
            
        aspect_ratios: tuple of floats
            Array of floats indicating the aspect raios of anchor boxes
            
        stride : int
            Output stride of the network

        """
        
        self.input_image_size = input_image_size
        self.anchor_areas = anchor_areas
        self.aspect_ratios = aspect_ratios
        self.stride = stride
        
        self.feature_map_height, self.feature_map_width = compute_network_output_feature_map_size(input_image_size,
                                                                                                  stride=stride)
        
        self.number_of_anchors_per_cell = len(anchor_areas) * len(aspect_ratios)
        
        # Precomputing anchor boxes positions
        
        self.precompute_anchor_boxes(input_image_size)
        
        
    def get_anchor_boxes_sizes(self):
        """Function to compute all possible sizes of anchor boxes given aspect ratios
        and anchor boxes areas.
        
        Computes all pairs of ```anchor_areas``` and ```aspect_ratios``` parameters
        resulting in bounding boxes of various sizes. Overall, the number of boxes is
        equal to ```size(anchor_areas) * size(aspect_ratios) ```.

        Returns
        -------
        anchor_boxes_sizes : numpy array of floats of size (1, #anchor_sizes, 2)
            Array containing all possible #anchor_sizes sizes of anchor boxes.
            The dummy dimension is intended for broadcastable capability with
            the coordinates of all possible anchor boxes.
        """
        
        anchor_boxes_sizes = []
        
        for current_anchor_area in self.anchor_areas:
            
            for current_aspect_ratio in self.aspect_ratios:
                
                # Given:
                # aspect_ratio = w / h
                # anchor_area = w * h
                # To find:
                # w and h
                # w = sqrt( aspect_ratio * anchor_area ) = sqrt( (w*w*h) / h ) = sqrt(w*w) = w
                
                w = math.sqrt( current_aspect_ratio * current_anchor_area )
                h = current_anchor_area / w
                
                anchor_boxes_sizes.append((w, h))
        
        # Adding a dummy dimension here in order to easily use .repeat() later
        return np.expand_dims( np.asarray(anchor_boxes_sizes), axis=0 )
    
    
    def get_anchor_boxes_center_coordinates(self, input_size):
        """Function to compute all possible coordinates of anchor boxes
        given the input image size and the output stride of the network.
        
        Computes the coordinates of centers of bounding boxes of each element
        of feature map with respect to the input image coordinate system.
        
        Parameters
        ----------
        input_size : tuple of ints
            Tuple with height and width sizes of the image
        

        Returns
        -------
        anchor_coordinates_input : numpy array of floats of size (#anchors, 1, 2)
            Array containing coordinates of all anchor boxes.
        """
        
        
        feature_map_height, feature_map_width = compute_network_output_feature_map_size(input_size, stride=self.stride)

        meshgrid_width, meshgrid_height = np.meshgrid(range(feature_map_width), range(feature_map_height))

        # Getting coordinates of centers of all the grid cells of the feature map
        anchor_coordinates_feature_map = zip(meshgrid_height.flatten(), meshgrid_width.flatten())
        anchor_coordinates_feature_map = np.asarray( anchor_coordinates_feature_map )
        anchor_coordinates_feature_map = anchor_coordinates_feature_map + 0.5

        anchor_coordinates_input = anchor_coordinates_feature_map * self.stride
        
        return np.expand_dims( anchor_coordinates_input, axis=1 )
        
    
    def precompute_anchor_boxes(self, input_size):
        """Function that combines all the previous functions to compute all anchor boxes
        with their coordinates with respect to the input image's coordinate system.
        
        Number of anchor boxes can be computed as:
         ```size(anchor_areas) * size(aspect_ratios) * (input_height / stride) * (input_width / stride) ```
                
        Parameters
        ----------
        input_size : tuple of ints
            Tuple with height and width sizes of the image
        

        Returns
        -------
        anchor_boxes : torch.FloatTensor of size (#anchors, 4)
            Array all anchor boxes in center_xywh format.
        """
        
        anchor_boxes_sizes = self.get_anchor_boxes_sizes()
        anchor_boxes_center_coordinates = self.get_anchor_boxes_center_coordinates(input_size)
        
        anchor_boxes_sizes_number = anchor_boxes_sizes.shape[1]
        anchor_boxes_center_coordinates_number = anchor_boxes_center_coordinates.shape[0]
        
        anchor_boxes_center_coordinates_repeated = anchor_boxes_center_coordinates.repeat(anchor_boxes_sizes_number, axis=1)
        anchor_boxes_sizes_repeated =  anchor_boxes_sizes.repeat(anchor_boxes_center_coordinates_number, axis=0)
        
        anchor_boxes = np.dstack((anchor_boxes_center_coordinates_repeated, anchor_boxes_sizes_repeated))
        
        anchor_boxes = anchor_boxes.reshape((-1, 4))
        
        self.anchor_boxes = torch.FloatTensor( anchor_boxes ).clone()
    
    def encode(self, ground_truth_boxes_center_xywh, ground_truth_labels):
        """Function that computes the parametrized in a certain way ground truth
        value for each anchor box given the groundtruth bounding boxes and their
        classes.
        
        Parameters
        ----------
        ground_truth_boxes_center_xywh : torch.FloatTensor of size (N, 4)
            Tensor contains N groundtruth bounding boxes in a center_xywh format
        
        ground_truth_labels:  torch.LongTensor of size (N,)
            Tensor containing 

        Returns
        -------
        target_deltas : torch.FloatTensor of size (#anchors, 4)
            Contains parametrized in a certain way differences between
            anchor boxes coordinates and sizes and closest ground truth box.
            
        target_labels : torch.FloatTensor of size (#anchors,)
            Contains groundtruth class labels for each anchor box.
            The labels are determined based on the amount of intersection of the anchor
            box and the closest groundtruth bounding box.
        """
        
                
        # (N, 4)
        anchor_boxes_center_xywh = self.anchor_boxes

        # --- Conversion stage
        # Converting anchor boxes and groudtruth boxes into xyxy format
        # in order to compute the intersection over union later on

        anchor_boxes_xyxy = convert_bbox_center_xywh_tensor_to_xyxy(anchor_boxes_center_xywh)
        ground_truth_boxes_xyxy = convert_bbox_center_xywh_tensor_to_xyxy(ground_truth_boxes_center_xywh)

        # --- Matching stage
        # Computing intersection over union between all pairs of anchor boxes
        # and groundtruth boxes
        
        # (N, M)
        ious = compute_bboxes_ious(anchor_boxes_xyxy, ground_truth_boxes_xyxy)

        # Getting ground truth box with the biggest intersection for
        # each anchor box. -- we get ids here
        # (N,)
        anchor_boxes_best_groundtruth_match_ious, anchor_boxes_best_groundtruth_match_ids = ious.max(1)

        # Here we actually extract the relevant groundtruth for each anchor box
        groundtruth_boxes_center_xywh_best_match_anchorwise = ground_truth_boxes_center_xywh[anchor_boxes_best_groundtruth_match_ids]

        # --- Regressing stage

        delta_xy = (groundtruth_boxes_center_xywh_best_match_anchorwise[:,:2]-anchor_boxes_center_xywh[:,:2]) / anchor_boxes_center_xywh[:,2:]
        delta_wh = torch.log(groundtruth_boxes_center_xywh_best_match_anchorwise[:,2:]/anchor_boxes_center_xywh[:,2:])

        target_deltas = torch.cat((delta_xy, delta_wh), dim=1)

        # Accounting for the background here
        # TODO: add special handeling of +1 shifting classes -- we dont' need that 
        # for pascal as all classes ids don't have 0 there but there might be other
        # datasets where it is not the case
        target_labels = ground_truth_labels[anchor_boxes_best_groundtruth_match_ids] #+ 1
        
        # TODO: during testing the threshold of 0.5 seemed to be too strict,
        # some groundtruth boxes didn't have any matched anchor boxes

        target_labels[anchor_boxes_best_groundtruth_match_ious < 0.4] = 0

        ignore = (anchor_boxes_best_groundtruth_match_ious > 0.3) & (anchor_boxes_best_groundtruth_match_ious < 0.4)

        target_labels[ignore] = -1
        
        # Resizing so that it's easier to use with some models
        # TODO: the following piece of code can be probably
        # placed in a separate function.
        
        original_shape = [self.feature_map_height,
                          self.feature_map_width,
                          self.number_of_anchors_per_cell]
        
        target_labels_reshaped_back = target_labels.view(original_shape)
        
        # Tensor with coordinates has additional dim with 4 elements: x, y, height, width
        original_shape.append(4)
        
        # TODO: the rest of the code probably belongs better to the dataloader
        # and we probably have to move it there
        
        target_deltas_reshaped_back = target_deltas.view(original_shape)
        
        target_deltas_reshaped_back = target_deltas_reshaped_back.permute(2, 3, 0, 1).contiguous()
        
        target_labels_reshaped_back = target_labels_reshaped_back.permute(2, 0, 1).contiguous()
        
        return target_deltas_reshaped_back, target_labels_reshaped_back
    
    def decode(self, anchors_deltas, anchors_logits):
        """Function that converts the predicted delta values for anchor boxes
        into final prediction bounding boxes with associated classes.

        Parameters
        ----------
        anchors_deltas : torch.FloatTensor of size (H*W*ANCHOR_BOXES_PER_CELL, 4)
            Tensor containing delta values for each anchor predicted by some model.
            Make sure that values are aligned in the same way (H, W, anchor_boxes_per_cell, 4)

        anchors_logits:  torch.LongTensor of size (H*W*ANCHOR_BOXES_PER_CELL, NUMBER_OF_CLASSES)
            Tensor containing logits predicted for each anchor box by the model.
            Make sure that the input values are properly aligned similar to previous argument.

        Returns
        -------
        boxes : torch.FloatTensor of size (#boxes, 4)
            Returns predicted boxes in the center_xywh format.

        classes : torch.LongTensor of size (#boxes,)
            Returns predicted classes for each of the returned boxes
        """


        anchor_boxes = self.anchor_boxes.type_as(anchors_deltas)

        # (H*W*anchors_per_cell, number_of_classes)
        # Variable() because softmax work only with them and not Tensors
        anchors_probabilities = torch.nn.functional.softmax(Variable(anchors_logits), dim=1).data

        anchors_probabilities_argmaxed_score, anchors_probabilities_argmaxed_class = anchors_probabilities.max(1)

        # anchor boxes that were classified as a non-background -- meaning that
        # they have an intersection of at least 0.5 IOU with 
        active_anchor_boxes_indexes = torch.nonzero(anchors_probabilities_argmaxed_class > 0).squeeze()

        loc_xy = anchors_deltas[:,:2]
        loc_wh = anchors_deltas[:,2:]

        xy = loc_xy * anchor_boxes[:,2:] + anchor_boxes[:,:2]
        wh = loc_wh.exp() * anchor_boxes[:,2:]

        boxes_center_xywh = torch.cat([xy, wh], 1)

        # Convert to xyxy to perform non maximum suppression
        boxes_xyxy = torch.cat([xy-wh/2, xy+wh/2], 1)

        # Getting the coordinates of only active anchor boxes
        active_boxes_center_xywh = boxes_center_xywh[active_anchor_boxes_indexes, :]
        activated_anchor_boxes_xyxy = boxes_xyxy[active_anchor_boxes_indexes, :]
        activated_anchor_scores = anchors_probabilities_argmaxed_score[active_anchor_boxes_indexes]
        activated_anchor_classes = anchors_probabilities_argmaxed_class[active_anchor_boxes_indexes]

        suppressed_indexes = box_nms(activated_anchor_boxes_xyxy, activated_anchor_scores).type_as(anchors_probabilities_argmaxed_class)

        final_boxes_center_xywh = active_boxes_center_xywh[suppressed_indexes]
        final_boxes_classes = activated_anchor_classes[suppressed_indexes]

        return final_boxes_center_xywh, final_boxes_classes

    
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
    
    return np.ceil(feature_map_size).astype(np.int)


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