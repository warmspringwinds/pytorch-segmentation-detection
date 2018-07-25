
import os, sys
import datetime
from PIL import Image
from glob import glob

import torch.utils.data as data


def generate_unique_timestamp_name():
    
    datetime_obj = datetime.datetime.now()
    
    uniq_filename = str(datetime_obj.date()) + '-' + str(datetime_obj.time()).replace(':', '-').replace('.', '-')
    
    return uniq_filename


class SimpleDataset(data.Dataset):
    
    
    def __init__(self,
                 root=None,
                 train=True,
                 number_of_classes=2,
                 joint_transform=None):
        
        self.number_of_classes = number_of_classes
        self.joint_transform = joint_transform
        
        if root is None:
            
            if train:
                
                self.root = os.path.expanduser( '~/.pytorch-segmentation-detection/datasets/simple_dataset/train')
            else:
                
                self.root = os.path.expanduser( '~/.pytorch-segmentation-detection/datasets/simple_dataset/val')
                
        self.images_folder = os.path.join( self.root, 'images' )
        self.annotation_folder = os.path.join( self.root, 'annotations' )
        
        if not os.path.exists(self.root):
            
            os.makedirs(self.images_folder)
            os.makedirs(self.annotation_folder)
            
        # Searching for files with any extention
        self.annotations_filenames = glob(os.path.join(self.annotation_folder, '*.*') )
        
        self.images_filenames = []
        
        for annotation_filename in self.annotations_filenames:
            
            annotation_filename_basename = os.path.basename(annotation_filename)
            
            # Trying to be as general as possible -- we don't assume any specific
            # extention for images and annotations
            annotation_filename_without_ext = os.path.splitext(annotation_filename_basename)[0]
            image_filename_regex = annotation_filename_without_ext + '.*'
            image_filename_regex = os.path.join( self.images_folder, image_filename_regex )
            image_filename = glob(image_filename_regex)[0]
            
            self.images_filenames.append(image_filename)
    
    def __len__(self):
        
        return len(self.images_filenames)
    
    def __getitem__(self, index):
        
        image_filename = self.images_filenames[index]
        annotation_filename = self.annotations_filenames[index]
        
        image = Image.open(image_filename)
        annotation = Image.open(annotation_filename)
        
        if self.joint_transform is not None:
            
            image, annotation = self.joint_transform([image, annotation])
            
        return image, annotation
            
            
    def add_new_sample(self, image_annotation_pair):
        
        # Generating new name using timestamps
        unqique_filename = generate_unique_timestamp_name() + '.png'
        image_save_path = os.path.join( self.images_folder, unqique_filename )
        annotation_save_path = os.path.join( self.annotation_folder, unqique_filename )

        image, annotation = image_annotation_pair
        
        image.save(image_save_path)
        annotation.save(annotation_save_path)

        self.images_filenames.append(image_save_path)
        self.annotations_filenames.append(annotation_save_path)
    
    def __delitem__(self, index):
        
        image_filepath = self.images_filenames[index] 
        annotation_filepath = self.annotations_filenames[index] 
        
        del self.images_filenames[index]
        del self.annotations_filenames[index]
        
        os.remove(annotation_filepath)
        os.remove(image_filepath)
    
    
    def __setitem__(self, index, updated_annotation):
        
        annotation_filepath = self.annotations_filenames[index] 
        
        updated_annotation.save(annotation_filepath)
        
        
    