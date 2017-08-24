import sys
import os

import torch
import torch.utils.data as data

import numpy as np

# Reading video files
import imageio

import skimage.io as io

from ..utils.endovis_instrument import clean_up_annotation, merge_left_and_right_annotations


class EndovisInstrument(data.Dataset):

    
    CLASS_NAMES = ['background', 'manipulator', 'shaft', 'ambigious']
    
    # Urls of original pascal and additional segmentations masks
    URL = 'https://endovissub-instrument.grand-challenge.org/'
    
    
    def __init__(self,
                 root,
                 train=True,
                 joint_transform=None,
                 prepare_dataset=False,
                 split_mode=2):
        
        self.root = root
        
        self.joint_transform = joint_transform
        
        if prepare_dataset:
            
            self._prepare_dataset()
        
        
        # TODO: Create train/val split later
        # if train:
            
        #     self.img_anno_pairs = pascal_annotation_filename_pairs_train_val[0]
            
        # else:
            
        #     self.img_anno_pairs = pascal_annotation_filename_pairs_train_val[1]
            
            
        
    def __len__(self):
        
        return len(self.img_anno_pairs)
    
    def __getitem__(self, index):
        
        img_path, annotation_path = self.img_anno_pairs[index]
        
        _img = Image.open(img_path).convert('RGB')
        
        # TODO: maybe can be done in a better way
        _target = Image.open(annotation_path)

        if self.joint_transform is not None:
            _img, _target = self.joint_transform([_img, _target])
        
        return _img, _target
        
        
    def _prepare_dataset(self):
        """
        Creates a new folder with the name Processed in the root of the dataset
        where all the images and annotations are stored as plain jpg and png images.
        """

        annotation_folder_to_save = os.path.join(self.root, 'Processed/annotations')
        images_folder_to_save = os.path.join(self.root, 'Processed/images')


        annotation_save_template = os.path.join( annotation_folder_to_save, "{0:08d}.png" )
        images_save_template = os.path.join( images_folder_to_save, "{0:08d}.jpg" )

        # Creating folders to save all the images and annotations
        if not os.path.exists(annotation_folder_to_save):
            os.makedirs(annotation_folder_to_save)

        if not os.path.exists(images_folder_to_save):
            os.makedirs(images_folder_to_save)

        # Creating template to go through the datasets folders
        dataset_folder_template = "Training/Dataset{}"
        dataset_template = os.path.join(self.root, dataset_folder_template)

        image_number_offset = 0

        # We have overall 4 datasets
        for current_dataset_number in xrange(1, 5): 

            current_dataset_path = dataset_template.format(current_dataset_number)

            if current_dataset_number == 1:

                # First dataset has two vides with separate annotations for each tool
                left_annotation_video_filename = os.path.join(current_dataset_path, 'Left_Instrument_Segmentation.avi')
                right_annotation_video_filename = os.path.join(current_dataset_path, 'Right_Instrument_Segmentation.avi')
            else:

                # Other datasets have just one video with annotation
                annotation_video_filename = os.path.join(current_dataset_path, 'Segmentation.avi')


            # Each dataset has just one video and it has the same name
            images_video_filename = os.path.join(current_dataset_path, 'Video.avi')

            # Creating readers for each of our videos
            images_reader = imageio.get_reader(images_video_filename,  'ffmpeg')

            # Once again -- first dataset is an exception
            if current_dataset_number == 1:

                left_annotations_reader = imageio.get_reader(left_annotation_video_filename, 'ffmpeg')
                right_annotations_reader = imageio.get_reader(right_annotation_video_filename, 'ffmpeg')
            else:

                annotations_reader = imageio.get_reader(annotation_video_filename, 'ffmpeg')

            current_dataset_number_of_images = images_reader.get_length()


            for current_image_number in xrange(current_dataset_number_of_images):

                # We need to merge two separate annotation files in the first dataset
                if current_dataset_number == 1:
                    current_annotatio_left = left_annotations_reader.get_data(current_image_number)
                    processed_current_annotation_left = clean_up_annotation(current_annotatio_left)

                    current_annotation_right = right_annotations_reader.get_data(current_image_number)
                    processed_current_annotation_right = clean_up_annotation(current_annotation_right)

                    processed_current_annotation_final = merge_left_and_right_annotations(processed_current_annotation_left,
                                                                                          processed_current_annotation_right)
                else:

                    current_annotation = annotations_reader.get_data(current_image_number)
                    processed_current_annotation_final = clean_up_annotation(current_annotation)

                current_image = images_reader.get_data(current_image_number)

                # add offset so that we respect the global count and not of the current dataset
                current_annotation_name_to_save = annotation_save_template.format(current_image_number + image_number_offset)
                current_image_name_to_save = images_save_template.format(current_image_number + image_number_offset)


                # add the offset from previous dataset image files -- so that we get all images saved
                io.imsave(current_annotation_name_to_save, processed_current_annotation_final)
                io.imsave(current_image_name_to_save, current_image)

            # Update the global count of images
            image_number_offset += current_dataset_number_of_images
        
    
    
        