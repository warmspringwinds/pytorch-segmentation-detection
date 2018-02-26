import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

from torchvision.datasets import CocoDetection

from ...utils.detection import (compute_bboxes_ious,
                                convert_bbox_topleft_xywh_tensor_to_center_xywh,
                                convert_bbox_center_xywh_tensor_to_xyxy,
                                convert_bbox_center_xywh_tensor_to_topleft_xywh,
                                display_bboxes_center_xywh,
                                compute_network_output_feature_map_size,
                                AnchorBoxesManager,
                                pad_to_size_with_bounding_boxes,
                               random_crop_with_bounding_boxes)


class PascalVOCDetection(data.Dataset):
    """
    PascalVOCDetection class serves as a wrapper for PASCAL VOC detection
    dataset -- delivering the target values in the parametrized representation
    which can be easily feeded into the model and later on into the loss function.
    
    It automatically manages the anchor boxes for an image of predefined size and delivers
    parametrized values for each anchor box for each sample.
    
    .json coco-like annotation file for pascal found in this thread:
    https://github.com/matterport/Mask_RCNN/issues/51
    Direct download link:
    https://storage.googleapis.com/coco-dataset/external/PASCAL_VOC.zip
    """
    
    def __init__(self, images_folder_path,
                 annotation_json,
                 image_transform,
                 input_image_size=(600, 600)
                ):
        """Constructor function for the PascalVOCDetection class.
        
        Given the images folder path, coco-like annotation file for pascal voc
        and image size, it automatically pads input images, parametrizes associated
        groundtruth boxes and classes. As, all images are padded to the same size,
        this gives us a chance to batch input images and associated parametrized groundtruth.
        
        Parameters
        ----------
        images_folder_path : string
            Global or relative path to pascal voc images folder.
            
        annotation_json : string
            Global or relative path to coco-like pascal annotation file (see above for download link).
            
        image_transform: torchvision.transforms object for preprocessing
            Stack of preprocessing functions that are being run on image
            
        input_image_size : tuple of ints
            Size of the all images that are being delivered -- we padd all of them to the
            same size (see above).

        """
        
        self.input_size = input_image_size
        self.images_folder_path = images_folder_path
        self.annotation_json = annotation_json
        
        self.image_transform = image_transform
        
        self.anchor_box_manager = AnchorBoxesManager(input_image_size=input_image_size)
        
        self.pascal_cocolike_db = CocoDetection(annFile=annotation_json,
                                               root=images_folder_path)
        
    def __len__(self):
        
        return len(self.pascal_cocolike_db)
    
    def __getitem__(self, idx):
        
        pil_img, cocolike_detection_annotations = self.pascal_cocolike_db[idx]
        
        bboxes_locations_topleft_xywh = map(lambda cocolike_dict: cocolike_dict['bbox'], cocolike_detection_annotations)
        bboxes_classes = map(lambda cocolike_dict: cocolike_dict['category_id'], cocolike_detection_annotations)
        
        # Getting the xywh coordinates and converting them to xyxy
        bboxes_locations_topleft_xywh = torch.FloatTensor( bboxes_locations_topleft_xywh )
        ground_truth_boxes_center_xywh = convert_bbox_topleft_xywh_tensor_to_center_xywh( bboxes_locations_topleft_xywh )
        
        ground_truth_labels = torch.LongTensor( bboxes_classes )
        
        #pil_img_padded, ground_truth_boxes_center_xywh_padded = pad_to_size_with_bounding_boxes(input_img=pil_img,
        #                                                                                        size=self.input_size,
        #                                                                                        bboxes_center_xywh=ground_truth_boxes_center_xywh)
        
        pil_img_padded, ground_truth_boxes_center_xywh_padded = random_crop_with_bounding_boxes(input_img=pil_img,
                                                                                                crop_size=self.input_size,
                                                                                                bboxes_center_xywh=ground_truth_boxes_center_xywh)
                                                                                                
        
        target_deltas, target_classes = self.anchor_box_manager.encode(ground_truth_boxes_center_xywh=ground_truth_boxes_center_xywh_padded,
                                                                       ground_truth_labels=ground_truth_labels)
        
        img_tensor_transformed = self.image_transform(pil_img_padded)
        
        #return pil_img_padded, ground_truth_boxes_center_xywh_padded
        return img_tensor_transformed, target_deltas, target_classes