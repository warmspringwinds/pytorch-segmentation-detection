import glob
import sys, os
import pydicom
import numpy as np
import pandas as pd
import torch.utils.data as data

from ..utils.rle_mask_encoding import rle2mask


class LungSegmentation(data.Dataset):
    
    
    def __init__(self,
                 train_images_folder_path=None,
                 annotation_csv_file_path=None,
                 train=True,
                 joint_transform=None):
        
        self.joint_transform = joint_transform
                
        self.train_images_folder_path = train_images_folder_path
        
        
        images_filenames = sorted(glob.glob(train_images_folder_path + "/*/*/*.dcm"))
        
        train_portion = 0.8
        num_train = len(images_filenames)
        split = int(np.floor(train_portion * num_train))
        
        if train:
            
            self.images_filenames = images_filenames[:split]
            
        else:
            
            self.images_filenames = images_filenames[split:]
        
        
        self.annotation_df = pd.read_csv(annotation_csv_file_path)
        
            
    def __len__(self):
        
        return len(self.images_filenames)
    
    def __getitem__(self, index):
        
        image_filename = self.images_filenames[index]
        
        image = pydicom.dcmread(image_filename).pixel_array
        
        image_filename_stripped = image_filename.split('/')[-1][:-4]
        
        annotation_rle = self.annotation_df.loc[self.annotation_df['ImageId'] == image_filename_stripped][' EncodedPixels'].values[0]
        
        if annotation_rle != ' -1':
        
            annotation = rle2mask(annotation_rle, width=1024, height=1024).transpose()
        else:
            
            annotation = np.zeros((1024, 1024))
        
        if self.joint_transform is not None:
            
            image, annotation = self.joint_transform([image, annotation])
            
        return image, annotation