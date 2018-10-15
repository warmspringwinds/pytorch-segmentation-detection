import os
import os.path
import numpy as np
from PIL import Image
import torch.utils.data as data


class NYUv2Segmentation(data.Dataset):
    
    # so far only 13 classes are available, will add 40 like in the paper
    # by Long et al. later.
    
    # TODO: write a script to load all the training data automatically
    
    #    test images: http://www.doc.ic.ac.uk/~ahanda/nyu_test_rgb.tgz
    #    train images: http://www.doc.ic.ac.uk/~ahanda/nyu_train_rgb.tgz
    #    test labels: https://github.com/ankurhanda/nyuv2-meta-data/raw/master/test_labels_13/nyuv2_test_class13.tgz
    #    train labels: https://github.com/ankurhanda/nyuv2-meta-data/raw/master/train_labels_13/nyuv2_train_class13.tgz
    
    dataset_types = ['train', 'test']
    
    images_subfolder_name = 'images'
    annotations_subfolder_name = 'annotations'
    
    ignore_label = 255

    
    def __init__(self,
                 dataset_root,
                 dataset_type=0,
                 joint_transform=None):

        # dataset_type:
        # 0 - train
        # 1 - test
        
        # Used for remapping from original labels to training ones
        self.ordered_train_labels = np.asarray( [self.ignore_label] + range(13) )

        self.dataset_root = dataset_root
        self.joint_transform = joint_transform

        dataset_type_name = self.dataset_types[dataset_type]

        images_folder_path = os.path.join(dataset_root, dataset_type_name, self.images_subfolder_name)

        annotations_folder_path = os.path.join(dataset_root, dataset_type_name, self.annotations_subfolder_name)

        images_filenames = sorted(os.listdir(images_folder_path))
        annotations_filenames = sorted(os.listdir(annotations_folder_path))

        self.images_filenames = list(map(lambda x: os.path.join(images_folder_path, x), images_filenames))
        self.annotations_filenames = list(map(lambda x: os.path.join(annotations_folder_path, x), annotations_filenames))
            
    def __len__(self):

        return len(self.images_filenames)
    
    
    def __getitem__(self, index):

        img_path = self.images_filenames[index]
        annotation_path = self.annotations_filenames[index]

        _img = Image.open(img_path).convert('RGB')
        
        # TODO: maybe can be done in a better way
        _target = Image.open(annotation_path)
        
        _target_np = np.asarray(_target).copy()
        
        # https://stackoverflow.com/questions/8188726/how-do-i-do-this-array-lookup-replace-with-numpy
        _target_np = self.ordered_train_labels[_target_np].astype(np.uint8)
        
        _target = Image.fromarray(_target_np)

        if self.joint_transform is not None:

            _img, _target = self.joint_transform([_img, _target])

        return _img, _target
