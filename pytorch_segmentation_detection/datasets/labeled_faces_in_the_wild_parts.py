
import os, sys
from glob import glob
import numpy as np
from PIL import Image
import torch.utils.data as data




def get_matching_image_filepath_for_annotation(annotation_name):
    
    person_name = '_'.join( annotation_name.split('_')[:-1] )

    annotation_picture_number_and_extention = annotation_name.split('_')[-1]

    image_picture_number_and_extention = annotation_picture_number_and_extention.replace('ppm', 'jpg')

    image_name = '_'.join([person_name, image_picture_number_and_extention])
    
    image_filepath = os.path.join(person_name, image_name)

    return image_filepath

def convert_lfw_parts_annotation_to_standart(lfw_annotation):

    anno_new = np.zeros(lfw_annotation.shape[:2], dtype=np.uint8)

    for annotation_slice_number in xrange(lfw_annotation.shape[2]):

        annotation_slice = lfw_annotation[:, :, annotation_slice_number]
        anno_new[annotation_slice > 0] = annotation_slice_number
    
    return anno_new


class LabeledFacesInTheWildParts(data.Dataset):
    
    annotations_sub_folder = 'parts_lfw_funneled_gt_images'
    images_sub_folder = 'lfw_funneled'
    
    number_of_classes = 3
    
    def __init__(self,
                 dataset_root,
                 train=True,
                 joint_transform=None):
        
        self.dataset_root = dataset_root
        self.joint_transform = joint_transform
        
        annotations_path = os.path.join(dataset_root, self.annotations_sub_folder)
        images_path = os.path.join(dataset_root, self.images_sub_folder)

        self.annotations_filenames = glob(os.path.join(annotations_path, '*.ppm') )
        
        if train:
            
            self.annotations_filenames = self.annotations_filenames[200:]
        else:
            
            self.annotations_filenames = self.annotations_filenames[:200]

        self.images_filenames = []

        for annotation_filename in self.annotations_filenames:

            annotation_basename = os.path.basename(annotation_filename)

            image_relative_filepath = get_matching_image_filepath_for_annotation(annotation_basename)

            image_filepath = os.path.join(images_path, image_relative_filepath)

            self.images_filenames.append(image_filepath)

    def __len__(self):
        
        return len(self.images_filenames)
    
    def __getitem__(self, index):
        
        img_path = self.images_filenames[index]
        annotation_path = self.annotations_filenames[index]
        
        _img = Image.open(img_path).convert('RGB')
        
        # TODO: maybe can be done in a better way
        _target = Image.open(annotation_path)
        
        _target_np = np.asarray(_target)
        
        _target_np = convert_lfw_parts_annotation_to_standart(_target_np)
        
        _target = Image.fromarray(_target_np)

        if self.joint_transform is not None:

            _img, _target = self.joint_transform([_img, _target])
            
        return _img, _target