import random
import numbers
import collections
import numpy as np
from PIL import Image, ImageOps

import torch


def split_image_into_tiles(input_image, block_rows, block_cols):
    """
    Credit:
    https://stackoverflow.com/questions/16873441/form-a-big-2d-array-from-multiple-smaller-2d-arrays/16873755#16873755
    https://stackoverflow.com/questions/13990465/3d-numpy-array-to-2d/13990648#13990648
    """
    
    input_rows, input_cols = input_image.shape[:2]
        
    input_depth = input_image.shape[2] if input_image.ndim == 3 else 0
        
    # Compute how many blocks will fit along rows and cols axes
    block_cols_number_in_input = input_cols // block_cols
    block_rows_number_in_input = input_rows // block_rows
    
    overall_number_of_blocks = block_rows_number_in_input * block_cols_number_in_input

    # Reshaping doesn't change c-arrangement of elements.
    # Reshaping can be looked at like applying ravel() function
    # and then grouping that 1D array into requested shape.
    
    # So if we form our input image in the following shape (below the comment)
    # we will see that if we swap 1st and 2nd axes (or transpose).
    # Trasposing in this case can be looked at like as if we index first
    # along 2nd and then 1st axis. In case of simple 2D matrix transpose -- 
    # we traverse elemets down first and right second.
    
    if input_depth:
        
        tmp = input_image.reshape((block_rows_number_in_input,
                                   block_rows,
                                   block_cols_number_in_input,
                                   block_cols,
                                   input_depth))
    else:
        
        tmp = input_image.reshape((block_rows_number_in_input,
                                   block_rows,
                                   block_cols_number_in_input,
                                   block_cols))
        
    tmp = tmp.swapaxes(1, 2)
    
    if input_depth:
        
        tmp = tmp.reshape(( overall_number_of_blocks, block_rows, block_cols, input_depth ))
        
    else:
        
        tmp = tmp.reshape(( overall_number_of_blocks, block_rows, block_cols))
        
        
    return tmp


def pad_to_size(input_img, size, fill_label=0):
    """Pads image to the size with fill_label if the input image is smaller"""

    input_size = np.asarray(input_img.size)
    padded_size = np.asarray(size)

    difference = padded_size - input_size

    parts_to_expand = difference > 0

    expand_difference = difference * parts_to_expand

    expand_difference_top_and_left = expand_difference // 2

    expand_difference_bottom_and_right = expand_difference - expand_difference_top_and_left
    
    # Form the PIL config vector
    pil_expand_array = np.concatenate( (expand_difference_top_and_left,
                                        expand_difference_bottom_and_right) )
    
    processed_img = input_img
    
    # Check if we actually need to expand our image.
    if pil_expand_array.any():
        
        pil_expand_tuple = tuple(pil_expand_array)
        
        processed_img = ImageOps.expand(input_img, border=pil_expand_tuple, fill=fill_label)
    
    return processed_img


def crop_center_numpy(img, crop_size):
    
    crop_width, crop_height = crop_size
    
    img_height, img_width = img.shape
    
    start_width = img_width//2-(crop_width//2)
    
    start_height = img_height//2-(crop_height//2)
    
    return img[start_height:start_height+crop_height,start_width:start_width+crop_width]


def pad_to_fit_tiles_pil(image, tile_size):
    
    original_size_in_pixels = np.asarray(image.size)

    adjusted_size_in_tiles = np.ceil( original_size_in_pixels / float(tile_size) ).astype(np.int)

    adjusted_size_in_pixels = adjusted_size_in_tiles * tile_size

    adjusted_img = pad_to_size(image, adjusted_size_in_pixels)
    
    return adjusted_img, adjusted_size_in_pixels, adjusted_size_in_tiles


def convert_labels_to_one_hot_encoding(labels, number_of_classes):

    labels_dims_number = labels.dim()

    # Add a singleton dim -- we need this for scatter
    labels_ = labels.unsqueeze(labels_dims_number)
    
    # We add one more dim to the end of tensor with the size of 'number_of_classes'
    one_hot_shape = list(labels.size())
    one_hot_shape.append(number_of_classes)
    one_hot_encoding = torch.zeros(one_hot_shape).type(labels.type())
    
    # Filling out the tensor with ones
    one_hot_encoding.scatter_(dim=labels_dims_number, index=labels_, value=1)
    
    return one_hot_encoding.byte()


class ComposeJoint(object):
    
    def __init__(self, transforms):
        
        self.transforms = transforms

    
    def __call__(self, x):
        
        for transform in self.transforms:
            
            x = self._iterate_transforms(transform, x)
            
        return x
    
    
    def _iterate_transforms(self, transforms, x):
        """Credit @fmassa:
         https://gist.github.com/fmassa/3df79c93e82704def7879b2f77cd45de
        """
    
    
        if isinstance(transforms, collections.Iterable):

            for i, transform in enumerate(transforms):
                x[i] = self._iterate_transforms(transform, x[i])
        else:
            
            if transforms is not None:
                x = transforms(x)


        return x
    
    
class RandomHorizontalFlipJoint(object):
    
    def __call__(self, inputs):
        
        # Perform the same flip on all of the inputs
        if random.random() < 0.5:
            
            return map(lambda single_input:  ImageOps.mirror(single_input), inputs) 
        
        
        return inputs

    
class RandomScaleJoint(object):
    
    def __init__(self, low, high, interpolations=[Image.BILINEAR, Image.NEAREST]):
        
        self.low = low
        self.high = high
        self.interpolations = interpolations
    
    
    def __call__(self, inputs):
        
        ratio = random.uniform(self.low, self.high)
        
        def resize_input(input_interpolation_pair):
            
            input, interpolation = input_interpolation_pair
            
            height, width = input.size[0], input.size[1]
            new_height, new_width = (int(ratio * height), int(ratio * width))
            
            return input.resize((new_height, new_width), interpolation)
            
        return map(resize_input, zip(inputs, self.interpolations))


    

class RandomCropJoint(object):
    
    def __init__(self, crop_size, pad_values=[0, 255]):
        
        if isinstance(crop_size, numbers.Number):
            
            self.crop_size = (int(crop_size), int(crop_size))
        else:
            
            self.crop_size = crop_size
        
        self.pad_values = pad_values
        

    def __call__(self, inputs):
        
        
        def padd_input(img_pad_value_pair):
            
            input = img_pad_value_pair[0]
            pad_value = img_pad_value_pair[1]
            
            return pad_to_size(input, self.crop_size, pad_value)
        
        padded_inputs = map(padd_input, zip(inputs, self.pad_values))
        
        # We assume that inputs were of the same size before padding.
        # So they are of the same size after the padding
        w, h = padded_inputs[0].size
        
        th, tw = self.crop_size
        
        if w == tw and h == th:
            return padded_inputs

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        
        outputs = map(lambda single_input: single_input.crop((x1, y1, x1 + tw, y1 + th)), padded_inputs)
        
        return outputs
    
    

class CropOrPad(object):
    
    
    def __init__(self, output_size, fill=0):
        
        self.fill = fill
        self.output_size = output_size
        
    def __call__(self, input):
        
        input_size = input.size
        
        input_position = (np.asarray(self.output_size) // 2) - (np.asarray(input_size) // 2)

        output = Image.new(mode=input.mode,
                           size=self.output_size,
                           color=self.fill)
        
        output.paste(input, box=tuple(input_position))
        
        return output

    
class ResizeAspectRatioPreserve(object):
    
    
    def __init__(self, greater_side_size, interpolation=Image.BILINEAR):
        
        self.greater_side_size = greater_side_size
        self.interpolation = interpolation

    def __call__(self, input):
       
        w, h = input.size

        if w > h:

            ow = self.greater_side_size
            oh = int( self.greater_side_size * h / w )
            return input.resize((ow, oh), self.interpolation)

        else:

            oh = self.greater_side_size
            ow = int(self.greater_side_size * w / h)
            return input.resize((ow, oh), self.interpolation)
        

        
class Copy(object):
    
    
    def __init__(self, number_of_copies):
        
        self.number_of_copies = number_of_copies
        
    def __call__(self, input_to_duplicate):
        
        
        # Inputs can be of different types: numpy, torch.Tensor, PIL.Image
                
        duplicates_array = []
        
        if isinstance(input_to_duplicate, torch.Tensor):
            
            for i in xrange(self.number_of_copies):
                duplicates_array.append(input_to_duplicate.clone())
        else:
            
            for i in xrange(self.number_of_copies):
                duplicates_array.append(input_to_duplicate.copy())
            
        return duplicates_array
    

# Assumed to be run on torch.Tensor
class Split2D(object):
    """
    Splits the Tensor into 2D tiles along given two dimensions,
    and stacks them along specified new dimension. Mainly used to
    split input 2D image into nonintersecting tiles and stack them
    along batch dimension. Can be used when the whole image doesn't fit
    into the available GPU memory.
    """
    
    
    def __init__(self,
                 split_block_sizes=(128, 128),
                 split_dims=(1, 2),
                 stack_dim=0):
        
        self.split_block_sizes = split_block_sizes
        self.split_dims = split_dims
        self.stack_dim = stack_dim
    
    
    def __call__(self, tensor_to_split):
        
        split_2d = []
        
        split_over_first_dim = tensor_to_split.split(self.split_block_sizes[0],
                                                     dim=self.split_dims[0])

        for current_first_dim_split in split_over_first_dim:

            split_2d.extend(current_first_dim_split.split(self.split_block_sizes[1],
                                                          dim=self.split_dims[1]))
        
        res = torch.stack(split_2d, dim=self.stack_dim)
        
        return res
    
    
    # Helper functions for reverse() method
    def squeeze_for_tensor_list(self, list_of_tensors, dim):
    
        return map(lambda x: x.squeeze(dim), list_of_tensors)

    
    def squeeze_for_2D_tensor_list(self, list2D_of_tensors, dim):
    
        return map(lambda x: self.squeeze_for_tensor_list(x, dim), list2D_of_tensors)
    
    
    def reverse(self, tensor_to_unsplit, dims_sizes):
        
        # First we get separate rows
        separate_rows = torch.split(tensor_to_unsplit,
                                    split_size=dims_sizes[1],
                                    dim=self.stack_dim)

        
        # Split each rows into separate column elements
        tensor_list_2D = map(lambda x: torch.split(x, split_size=1, dim=self.stack_dim), separate_rows)
        
        # Remove singleton dimension, so that we can use original self.split_dims
        tensor_list_2D = self.squeeze_for_2D_tensor_list(tensor_list_2D, self.stack_dim)

        concatenated_columns = map(lambda x: torch.cat(x, dim=self.split_dims[1]), tensor_list_2D)
        
        unsplit_original_tensor = torch.cat(concatenated_columns, dim=self.split_dims[0])
        
        return unsplit_original_tensor