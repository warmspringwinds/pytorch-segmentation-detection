import os, sys, glob

import skimage.io as io
import numpy as np
import torch.utils.data as data
import copy


def crop_actual_surgical_image(image):
    
    # The surgical images contain blank regions which are not useful and
    # better be removed
    
    return image[28:1052, 320:1600, :]

# in order to efficiently download the dataset, install megadown bash downloader:
# https://github.com/tonikelope/megadown
# Useful for the cases when the server doesn't have gui where you can launch browser

def get_sorted_by_name_image_names(folder_path, image_ext="png"):
    
    # Names in the dataset can be lexigraphically sorted
    # because frames are named with leading zeros
    
    file_regex = "*." + image_ext
    
    full_path_regex = os.path.join(folder_path, file_regex)
    
    unsorted_file_names = glob.glob(full_path_regex)
    
    sorted_file_names = sorted(unsorted_file_names)
    
    return sorted_file_names


class Endovis_Instrument_2017(data.Dataset):
    
    number_of_datasets = 8
    
    sequence_subfolder_template = 'instrument_dataset_{}'
    
    images_dir_name = 'left_frames'
    
    annotations_dir_name = 'ground_truth'
    
    instrument_names = ['Needle Driver',
                        'Vessel Sealer',
                        'Curved Scissors',
                        'Prograsp Forceps',
                        'Bipolar Forceps',
                        'Grasping Retractor']

    # Mapping of instrument name to a folder in the 'groundtruth' subfolder with
    # respective groundtruth files. For each training sequence
    instrument_names_to_groundtruth_folder_mapping_template = [
            {
                'Bipolar Forceps': ['Maryland_Bipolar_Forceps_labels'],
                'Prograsp Forceps': ['Left_Prograsp_Forceps_labels', 'Right_Prograsp_Forceps_labels'],
            },

            {
                'Prograsp Forceps': ['Left_Prograsp_Forceps_labels', 'Right_Prograsp_Forceps_labels']
            },

            {
                'Needle Driver': ['Left_Large_Needle_Driver_labels', 'Right_Large_Needle_Driver_labels']
            },

            {
                'Needle Driver': ['Large_Needle_Driver_Left_labels', 'Large_Needle_Driver_Right_labels'],
                'Prograsp Forceps': ['Prograsp_Forceps_labels']
            },

            {
                'Bipolar Forceps': ['Bipolar_Forceps_labels'],
                'Grasping Retractor': ['Grasping_Retractor_labels'],
                'Vessel Sealer': ['Vessel_Sealer_labels']
            },

            {
                'Needle Driver': ['Left_Large_Needle_Driver_labels', 'Right_Large_Needle_Driver_labels'],
                'Curved Scissors': ['Monopolar_Curved_Scissors_labels'],
                'Prograsp Forceps': ['Prograsp_Forceps']
            },


            {
                'Bipolar Forceps': ['Left_Bipolar_Forceps'],
                'Vessel Sealer': ['Right_Vessel_Sealer']
            },

            {
                'Bipolar Forceps': ['Bipolar_Forceps_labels'],
                'Grasping Retractor': ['Left_Grasping_Retractor_labels', 'Right_Grasping_Retractor_labels'],
                'Curved Scissors': ['Monopolar_Curved_Scissors_labels']
            }
        ]
    
    def get_instrument_names_to_groundtruth_folder_mapping_with_fullpaths(self):
        
        instrument_names_to_groundtruth_folder_mapping = copy.deepcopy(self.instrument_names_to_groundtruth_folder_mapping_template)
        
        for dataset_number in xrange(0, self.number_of_datasets):

            # Changes the name depending on the dataset number
            dataset_subfolder_name = self.sequence_subfolder_template.format(dataset_number + 1)

            dataset_path = os.path.join(self.root, dataset_subfolder_name)

            dataset_annotations_path = os.path.join(dataset_path, self.annotations_dir_name)

            dataset_annotations_dict = instrument_names_to_groundtruth_folder_mapping[dataset_number]

            for instrument_type in dataset_annotations_dict:

                current_instrument_full_groundtruth_foldernames = map(lambda x: os.path.join(dataset_annotations_path, x),
                                                                      dataset_annotations_dict[instrument_type] )

                dataset_annotations_dict[instrument_type] = current_instrument_full_groundtruth_foldernames
                
        
        return instrument_names_to_groundtruth_folder_mapping
    
    
    def __init__(self,
                 root,
                 train=True,
                 joint_transform=None,
                 validation_dataset_number_list=[2]):
        
        self.root = root
        
        self.joint_transform = joint_transform
        
        self.instrument_names_to_groundtruth_folder_mapping = self.get_instrument_names_to_groundtruth_folder_mapping_with_fullpaths()
        
        # Training datasets are all datasets
        # that are not in validation set
        training_dataset_number_list = []
        
        for dataset_number in xrange(self.number_of_datasets):
            
            if( dataset_number not in validation_dataset_number_list ):
                
                training_dataset_number_list.append(dataset_number)
                
        if train:
            
            self.img_annotations_filenames_tuples = self.get_datasets_filenames(training_dataset_number_list)
            
        else:
            
            self.img_annotations_filenames_tuples = self.get_datasets_filenames(validation_dataset_number_list)
    
    
    def get_single_dataset_filenames(self, dataset_number):
        
        # Changes the name depending on the dataset number
        dataset_subfolder_name = self.sequence_subfolder_template.format(dataset_number + 1)

        dataset_path = os.path.join(self.root, dataset_subfolder_name)

        # Get paths to images + different annotations
        dataset_images_path = os.path.join(dataset_path, self.images_dir_name)

        dataset_sorted_image_names = get_sorted_by_name_image_names(dataset_images_path)

        current_names_to_groundtruth_mapping = self.instrument_names_to_groundtruth_folder_mapping[dataset_number]

        overall_list = []

        for instrument_name in current_names_to_groundtruth_mapping:

            current_left_right_foldernames = current_names_to_groundtruth_mapping[instrument_name]

            current_left_right_filenames = map(lambda x: get_sorted_by_name_image_names(x), current_left_right_foldernames)

            current_number_of_frames = len(current_left_right_filenames[0])

            # Create the overall list if it hasn't already been created

            if not overall_list:

                overall_list = [{} for i in xrange(current_number_of_frames)]

            # Zip the left and right filenames into tuples
            # In case when there is just one instrument of a particular type,
            # we will have a tuple of size one
            # We also convert the tuple to lists outright using map
            current_instrument_annotation_filenames = map(list, zip(*current_left_right_filenames))

            for index, current_dict in enumerate(overall_list):

                current_dict[instrument_name] = current_instrument_annotation_filenames[index]


        return zip(dataset_sorted_image_names, overall_list)
    
    
    def get_datasets_filenames(self, dataset_number_list):
        
        all_filenames = []
        
        for current_dataset_number in dataset_number_list:
            
            current_dataset = self.get_single_dataset_filenames(current_dataset_number)
            
            all_filenames.extend(current_dataset)
        
        return all_filenames
    
    