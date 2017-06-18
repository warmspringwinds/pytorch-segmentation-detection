import random
import collections
from PIL import Image


class ComposeJoint(object):
    
    def __init__(self, transforms):
        
        self.transforms = transforms

    
    def __call__(self, x):
        
        for transform in self.transforms:
            
            x = self._iterate_transforms(transform, x)
            
        return x
    
    
    def _iterate_transforms(self, transforms, x):
        "Credit @fmassa.
         https://gist.github.com/fmassa/3df79c93e82704def7879b2f77cd45de
        "
    
    
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