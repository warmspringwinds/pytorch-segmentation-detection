import random
import numbers
import collections
import numpy as np
from PIL import Image, ImageOps



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
            
            return map(lambda single_input:  single_input.transpose(Image.FLIP_LEFT_RIGHT), inputs) 
        
        
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