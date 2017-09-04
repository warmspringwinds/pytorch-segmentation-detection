import torch
import torch.utils.data as data

import numpy as np

from six.moves import urllib
from PIL import Image
import tarfile
import os, sys

from ..utils.pascal_voc import get_augmented_pascal_image_annotation_filename_pairs
from ..utils.pascal_voc import convert_pascal_berkeley_augmented_mat_annotations_to_png


class PascalVOCSegmentation(data.Dataset):

    
    CLASS_NAMES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                   'dog', 'horse', 'motorbike', 'person', 'potted-plant',
                   'sheep', 'sofa', 'train', 'tv/monitor', 'ambigious']
    
    # Urls of original pascal and additional segmentations masks
    PASCAL_URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
    BERKELEY_URL = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz'
    
    PASCAL_TAR_FILENAME = "VOCtrainval_11-May-2012.tar"
    BERKELEY_TAR_FILENAME = "benchmark.tgz"
    
    PASCAL_ROOT_FOLDER_NAME = "VOCdevkit"
    BERKELEY_ROOT_FOLDER_NAME = "benchmark_RELEASE"
    
    
    def __init__(self,
                 root,
                 train=True,
                 joint_transform=None,
                 download=False,
                 split_mode=2):
        
        self.root = root
        self.pascal_tar_full_download_filename = os.path.join(self.root, self.PASCAL_TAR_FILENAME)
        self.berkeley_tar_full_download_filename = os.path.join(self.root, self.BERKELEY_TAR_FILENAME)
        
        self.pascal_full_root_folder_path = os.path.join(self.root, self.PASCAL_ROOT_FOLDER_NAME)
        self.berkeley_full_root_folder_path = os.path.join(self.root, self.BERKELEY_ROOT_FOLDER_NAME)
        
        self.joint_transform = joint_transform
        
        if download:
            
            self._download_dataset()
            self._extract_dataset()
            self._prepare_dataset()
        
        
        pascal_annotation_filename_pairs_train_val = get_augmented_pascal_image_annotation_filename_pairs(
                                                          self.pascal_full_root_folder_path,
                                                          self.berkeley_full_root_folder_path,
                                                          mode=split_mode)
        if train:
            
            self.img_anno_pairs = pascal_annotation_filename_pairs_train_val[0]
            
        else:
            
            self.img_anno_pairs = pascal_annotation_filename_pairs_train_val[1]
            
            
        
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
        
        
    def _download_dataset(self):
        
        
        # Add a progress bar for the download
        def _progress(count, block_size, total_size):
        
            progress_string = "\r>> {:.2%}".format( float(count * block_size) / float(total_size) )  
            sys.stdout.write(progress_string)
            sys.stdout.flush()
        
        # Create the root folder with all the intermediate
        # folders if it doesn't exist yet.
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        
        
        # TODO: factor this out into separate function because we repeat
        # same operation two times
        if os.path.isfile(self.pascal_tar_full_download_filename):
            
            print('\n PASCAL VOC segmentation dataset file already exists')
        else:
            
            print("\n Downloading PASCAL VOC segmentation dataset to {}".format(self.pascal_tar_full_download_filename))
            urllib.request.urlretrieve(self.PASCAL_URL, self.pascal_tar_full_download_filename, _progress)
            
            
        if os.path.isfile(self.berkeley_tar_full_download_filename):
            
            print('\n Berkeley segmentation dataset file already exists')
        else:
            
            print("\n Downloading Berkeley segmentation additional dataset to {}".format(self.berkeley_tar_full_download_filename))
            urllib.request.urlretrieve(self.BERKELEY_URL, self.berkeley_tar_full_download_filename, _progress)
        
    
    def _extract_tar_to_the_root_folder(self, tar_full_filename):
        
        # TODO: change to with: statement instead
        tar_obj = tarfile.open(tar_full_filename)
        
        tar_obj.extractall(path=self.root)
        
        tar_obj.close()
    
    def _extract_dataset(self):
        
        print("\n Extracting PASCAL VOC segmentation dataset to {}".format(self.pascal_full_root_folder_path))
        self._extract_tar_to_the_root_folder(self.pascal_tar_full_download_filename)
        
        print("\n Extracting Berkeley segmentation dataset to {}".format(self.berkeley_full_root_folder_path))
        self._extract_tar_to_the_root_folder(self.berkeley_tar_full_download_filename)
        
    def _prepare_dataset(self):
        
        print("\n Converting .mat files in the Berkeley dataset to pngs")
        
        convert_pascal_berkeley_augmented_mat_annotations_to_png(self.berkeley_full_root_folder_path)
        