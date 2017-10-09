import numpy as np
import skimage.morphology

#TODO: write good docs for these functions

def clean_up_annotation(annotation, labels=np.asarray([0, 70, 160]), ambigious_class_id=255):
    """Cleans up the artifacts of the segmentation mask caused by codec.
       In addition to that, relabels the annotations to be sequential
       number starting from 0. For example, [0, 70, 160] will be
       converted to [0, 1, 2]
    """

    # Nearest neighbour based on the distance to the class number
    labels_matrix = labels.reshape((1, 1, 3))
    distance_map = np.abs( annotation - labels_matrix )
    closest_neighbour_map = distance_map.argmin(axis=2)


    # morphology.label finds connected components.
    # we have to do it separately for each channel, because if
    # we run it on image with all labels, it will be hard to 
    # get the mapping from old lables to new ones (at least we didn't found a way to do it)


    # Do this for all classes except background.
    # Algorithm finds the biggest connected component, other components are
    # marked as ambigious (255). This is based on prior knowledge that shaft
    # and manipulator labels are single connected components in each training frame.
    for current_class_number in xrange(1, len(labels)):

        current_class_binary_mask = (closest_neighbour_map == current_class_number)

        # Find all connected components for current class
        res = skimage.morphology.label(current_class_binary_mask)

        # Get the number and number of elements in each connected component
        unique_labels, inverse, unique_counts = np.unique(res, return_inverse=True, return_counts=True)

        # unique_labels are sorted -- so the background label (represented by 0)
        # is always at the beginning of the list
        # but we are interested in connected component which is not a background
        # so we don't consider background to be a connected component
        not_background_classes = unique_labels[1:]
        not_background_classes_element_counts = unique_counts[1:]

        class_with_biggest_count = not_background_classes[not_background_classes_element_counts.argmax()]

        # Use the biggest component as ground truth mask
        # all other components will be marked as ambigious (255)
        closest_neighbour_map[res == class_with_biggest_count] = current_class_number
        closest_neighbour_map[(res != class_with_biggest_count) & (res != 0)] = ambigious_class_id
    
    return closest_neighbour_map


def merge_left_and_right_annotations(left_annotation,
                                     right_annotation,
                                     labels=np.asarray([0, 70, 160]),
                                     ambigious_class_id=255):
    # A function was written only for the case when tools doesn't intesect which
    # is the case in the first video.
    
    final_annotation = left_annotation.copy()

    # forgot the ambigious class 
    for i in range(1, len(labels)) + [ambigious_class_id]:

        union_mask = (right_annotation == i) | (left_annotation == i)

        final_annotation[union_mask] = i
        
    return final_annotation


def merge_left_and_right_annotations_v2(left_annotation,
                                        right_annotation,
                                        labels=np.asarray([0, 70, 160]),
                                        ambigious_class_id=255):
    
    # A function was written only for the case when tools doesn't intesect which
    # is the case in the first video.
    
    final_annotation = left_annotation.copy()

    # forgot the ambigious class 
    for i in labels + [ambigious_class_id]:

        union_mask = (right_annotation == i) | (left_annotation == i)

        final_annotation[union_mask] = i
        
    return final_annotation