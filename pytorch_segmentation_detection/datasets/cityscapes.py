
import os
import os.path
import numpy as np
from PIL import Image
import torch.utils.data as data

from ..utils.cityscapes import labels as cityscapes_labels


class Cityscapes(data.Dataset):
    
    # Images name subfolder in the root folder of the dataset
    images_subfolder = 'leftImg8bit'
    
    # Annotation name subfolder in the root folder of the dataset
    annotations_subfolder = 'gtFine'
        
    # Names of the folders containing train/val/test splits
    dataset_types = ['train', 'val', 'test']
    
    # Train labels which are used to map the labels
    # in the annotation images into train labels
    # Some variables in the annotatio are ignore for example
    # See utils.cityscapes for more details
    ordered_train_labels = np.asarray( map(lambda x: x.trainId, cityscapes_labels) )
    
    number_of_classes = 19

    
    def __init__(self,
                 dataset_root,
                 dataset_type=0,
                 train=True,
                 joint_transform=None):
        
        # dataset_root should point to a folder
        # with gtFine and leftImg8bit folders containing
        # annotations and images respectively.
        
        # dataset_type:
        # 0 - train
        # 1 - val
        # 2 - test
        
        self.dataset_root = dataset_root
        self.joint_transform = joint_transform

        dataset_type_name = self.dataset_types[dataset_type]

        images_folder_path = os.path.join(dataset_root, self.images_subfolder, dataset_type_name)
        annotations_folder_path = os.path.join(dataset_root, self.annotations_subfolder, dataset_type_name)

        self.images_filenames = []
        self.annotations_filenames = []

        for dirpath, dirnames, filenames in os.walk(images_folder_path):

            for filename in filenames:

                image_filename = os.path.join(dirpath, filename)

                annotation_filename = os.path.join( dirpath.replace(images_folder_path, annotations_folder_path),
                                                    filename.replace('leftImg8bit', 'gtFine_labelIds') )

                self.images_filenames.append( image_filename )
                self.annotations_filenames.append( annotation_filename )

        
        
    def __len__(self):

        return len(self.images_filenames)
    

    def __getitem__(self, index):

        img_path = self.images_filenames[index]
        annotation_path = self.annotations_filenames[index]

        _img = Image.open(img_path).convert('RGB')
        
        # TODO: maybe can be done in a better way
        _target = Image.open(annotation_path)
        
        _target_np = np.asarray(_target)
        
        # https://stackoverflow.com/questions/8188726/how-do-i-do-this-array-lookup-replace-with-numpy
        _target_np = self.ordered_train_labels[_target_np].astype(np.uint8)
        
        _target = Image.fromarray(_target_np)

        if self.joint_transform is not None:

            _img, _target = self.joint_transform([_img, _target])

        return _img, _target