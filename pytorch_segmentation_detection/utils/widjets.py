
from matplotlib import pyplot as plt
from ipywidgets import interactive
from ipywidgets import IntSlider
from PIL import Image
import numpy as np
import math


# --- Public functions

def create_dataset_paginated_display_widget(dataset_obj, number_of_images_per_page=5, ignore_label=255):
    """Creates a paginated widjet object for a pytorch.dataset object.
    
    Example:
    
    (assuming you are running in the jupyter notebook)
    
    %matplotlib inline
    
    from IPython.display import display
    
    widjet = create_dataset_paginated_display_widget(trainset)

    display(widjet)
    
    (will display an interactive paginated representation of your dataset)
    
    Parameters
    ----------
    dataset_obj : data.Dataset object
        Dataset object
        
    number_of_images_per_page : int
        Number of images and respective groundtruths to
        display per page
        
    ignore_label: int
        A lable that is used to represent regions to be ignored
        during training. 
        
    Returns
    -------
    widjet : object created by ipywidgets.interactive
        Widjet that can be displayed later on in jupyter notebook
    """
    
    number_of_images = len( dataset_obj )
    number_of_pages = int( math.ceil( number_of_images / number_of_images_per_page ) )
    
    def display_page_callback(page_number):
    
        for current_image_number in xrange(number_of_images_per_page):

            current_global_index = page_number * number_of_images_per_page + current_image_number

            if current_global_index > number_of_images-1:
                break

            dataset_sample_tuple = dataset_obj[current_global_index]
            
            tuple_size = len(dataset_sample_tuple)
        
            f, axies_list = plt.subplots(1, tuple_size, figsize=(20, 10))
            
            sample_element_axis_pairs = zip(dataset_sample_tuple, axies_list)
            
            for index, sample_axis_pair in enumerate(sample_element_axis_pairs):
                
                bind_axis_and_data(sample_axis_pair, index)
        
    
    slider_obj = IntSlider(min=0, max=number_of_pages-1, continuous_update=False)

    widjet = interactive( display_page_callback, page_number=slider_obj, continuous_update=False )
    
    return widjet

# --- Internal functions

def make_annotation_display_friendly(annotation, ignore_label=255):
    """Updates segmentation annotation so that labels become more
    distinguishable while displayed. Ignore is changed to become 0.
     
    Parameters
    ----------
    annotation : ndarray of ints
        Output shape of samples
    ignore_label : int
        Label representing the pixels to be ignored
    
    Returns
    -------
    annotation : ndarray of ints
        Updated annotation array
    """
    
    annotation = annotation + 1
    annotation[annotation == ignore_label] = 0
    # spreading labels so that they become more far apart from each other
    # TODO: won't be able to handle a large number of classes,
    # we need to adjust it for that
    annotation = annotation * 10

    return annotation

def bind_axis_and_data(data_axis_pair, position):
    """Updates segmentation annotation so that labels become more
     distinguishable while displayed. Ignore is changed to become 0.
    
    Parameters
    ----------
    data_axis_pair : tuple
        Tuple containing data (int or PIL.Image) which reresents groundtruth
        segmentation or label and an matplotlib's axis object which is supposed
        to be used to display this data.
    postion : int
        Position of the displayed data in the frame (starting from the left)
    """
    
    data, axis = data_axis_pair
    
    # Turning of the border of the plot
    axis.axis('off')
    
    if isinstance(data, int):
        
        # If it's a class number, draw it as image
        axis.annotate(str(data), xy=(0.5, 0.5), fontsize=50)
        
        return
        
    if not isinstance(data, Image.Image):
        
        return
    
    # If the script reaches until here, we are working with PIL image
    # Converting it to numpy first
    data = np.asarray(data)
    
    # If it is a segmentatio annotation image, adjust the
    # color to properly display
    if position != 0:
        
        data = make_annotation_display_friendly(data)
        axis.imshow(data, cmap="tab20b")
        return
        
    axis.imshow(data)
