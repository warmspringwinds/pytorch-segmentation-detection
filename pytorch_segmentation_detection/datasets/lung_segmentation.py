import glob
import sys, os
import pydicom
import numpy as np
import pandas as pd
from PIL import Image
import torch.utils.data as data
import random

from ..utils.rle_mask_encoding import rle2mask



def merge_masks_of_duplicated_images(annotation_df):
    
    annotation_df_without_duplicates = annotation_df.drop_duplicates(subset='ImageId', keep='first')

    res = annotation_df[annotation_df.duplicated(['ImageId'], keep=False)]

    unique_names = res['ImageId'].unique()

    for unique_name in unique_names:

        all_masks_rows = res[res['ImageId'] == unique_name]

        merged_mask_np = None

        for index, mask_row in all_masks_rows.iterrows():

            mask_np = rle2mask( mask_row[" EncodedPixels"], width=1024, height=1024 )

            if merged_mask_np is None:

                merged_mask_np = mask_np
            else:
                merged_mask_np[mask_np != 0] = 255

        merged_mask_rle = mask2rle(merged_mask_np, width=1024, height=1024)

        annotation_df_without_duplicates.loc[annotation_df_without_duplicates['ImageId'] == unique_name, [' EncodedPixels']] = merged_mask_rle
    
    return annotation_df_without_duplicates



class LungSegmentation(data.Dataset):
    
    
    def __init__(self,
                 train_images_folder_path=None,
                 annotation_csv_file_path=None,
                 train=True,
                 joint_transform=None):
        
        self.joint_transform = joint_transform
        
        self.train = train
                
        self.train_images_folder_path = train_images_folder_path
        
        self.annotation_df = pd.read_csv(annotation_csv_file_path)
        
        images_filenames = sorted(glob.glob(train_images_folder_path + "/*/*/*.dcm"))
        
        trancuted_name_and_full_names_lookup_dict = {}

        for images_filename in images_filenames:

            trancuted_name_and_full_names_lookup_dict.update({images_filename.split('/')[-1][:-4]: images_filename})
        
        if train:
        
            # Overall 8296
            negative_examples = self.annotation_df.loc[self.annotation_df[' EncodedPixels'] == ' -1'][250:]['ImageId'].unique()

            # Overall 2379
            posititve_examples = self.annotation_df.loc[self.annotation_df[' EncodedPixels'] != ' -1'][250:]['ImageId'].unique()
        else:
            
            # Overall 8296
            negative_examples = self.annotation_df.loc[self.annotation_df[' EncodedPixels'] == ' -1'][:250]['ImageId'].unique()

            # Overall 2379
            posititve_examples = self.annotation_df.loc[self.annotation_df[' EncodedPixels'] != ' -1'][:250]['ImageId'].unique()

        all_data = list(negative_examples) + list(posititve_examples)
        
        final = []

        for truncated_name in all_data:

            final.append(trancuted_name_and_full_names_lookup_dict[truncated_name])
            
#         if train:
            
#             negative_examples = list(negative_examples)
#             positive_examples = list(posititve_examples)
            
#             self.negative_examples = []
#             self.positive_examples = []
            
#             for truncated_name in negative_examples:
                
#                     self.negative_examples.append(trancuted_name_and_full_names_lookup_dict[truncated_name])
            
#             for truncated_name in positive_examples:
                
#                     self.positive_examples.append(trancuted_name_and_full_names_lookup_dict[truncated_name])
            
            
        
        self.images_filenames = final
        
            
    def __len__(self):
        
        return len(self.images_filenames)
    
    def __getitem__(self, index):
        
        image_filename = self.images_filenames[index]
        
#         if self.train:
            
#             if random.random() < 0.5:
                
#                 image_filename = random.choice(self.negative_examples)
#             else:
                
#                 image_filename = random.choice(self.positive_examples)
       
        image = pydicom.dcmread(image_filename).pixel_array
        
        image = np.dstack((image, image, image))
        
        #print(image.shape)

        image_filename_stripped = image_filename.split('/')[-1][:-4]

        annotation_rle = self.annotation_df.loc[self.annotation_df['ImageId'] == image_filename_stripped][' EncodedPixels'].values[0]

        if annotation_rle != ' -1':

            annotation = rle2mask(annotation_rle, width=1024, height=1024).transpose()
            annotation[annotation != 0] = 1
        else:

            annotation = np.zeros((1024, 1024))
        
        
        
        image = Image.fromarray(image)
        annotation = Image.fromarray(annotation)

        if self.joint_transform is not None:

            image, annotation = self.joint_transform([image, annotation])
            
        return image, annotation