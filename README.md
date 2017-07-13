# Dense-ai: Image Segmentation and Object Detection library

```Dense-ai``` is a library for dense inference and training of Convolutional Neural Networks (CNNs) on Images.
The aim of the library is to provide/provide a simplified way to:

- Converting some popular general/medical/other Image Segmentation and Detection Datasets into easy-to-use for training
format (Pytorch's dataloader).
- Training routine with on-the-fly data augmentation (scaling, color distortion).
- Training routine that is proved to work for particular model/dataset pair.
- Evaluating Accuracy of trained models with common accuracy measures: Mean IOU, Mean pix. accuracy, Pixel accuracy, Mean AP.
- Model files that were trained on a particular dataset with reported accuracy (models that were trained using
this library with reported training routine and not models that were converted from Caffe or other framework)
- Model definitions (like FCN-32s and others) that use weights initializations from Image Classification models like
VGG that are officially provided by ```Pytorch/Vision``` library.

So far, the library contains an implementation of FCN-32s (Long et al.), Resnet-18-8s, Resnet-34-8s (Chen et al.) image segmentation models in ```Pytorch``` and ```Pytorch/Vision``` library with training routine, reported accuracy,
trained models for PASCAL VOC 2012 dataset. To train these models on your data, you will have
to write a ```dataloader``` for your dataset.

Models for Object Detection will be released soon.


## Installation

This code requires:

1. [Pytorch](https://github.com/pytorch/pytorch).

2. Our [pytorch/vision](https://github.com/pytorch/vision) fork, which might be [merged](https://github.com/pytorch/vision/pull/184) and [futher merged](https://github.com/pytorch/vision/pull/190) in a future.

 Simply run:
 
 ```git clone -b fully_conv_resnet https://github.com/warmspringwinds/vision```
 
 And add ```pytorch/vision``` subdirectory to your path:

 ```python
 import sys
 # update with your path
 sys.path.insert(0, 'your/path/vision/')
 ```
3. Some libraries which can be acquired by installing [Anaconda package](https://www.continuum.io/downloads).
 
 Or you can install ```scikit-image```, ```matplotlib```, ```numpy``` using ```pip```.
 
5. Clone this library:

 ```git clone https://github.com/warmspringwinds/dense-ai```
 
 And add it to the path:
 
 ```python
 import sys
 # update with your path
 sys.path.append("/your/path/dense-ai/")
 ```


## PASCAL VOC 2012

Implemented models were tested on Restricted PASCAL VOC 2012 Validation dataset (RV-VOC12) and trained on
the PASCAL VOC 2012 Training data and additional Berkeley segmentation data for PASCAL VOC 12.
It was important to test models on restricted Validation dataset to make sure no images in the
validation dataset were seen by model during training.

The code to acquire the training and validating the model is also provided in the library.

### Fully Convolutional Networks for Semantic Segmentation (FCNs)

Here you can find models that were described in the paper "Fully Convolutional Networks for Semantic Segmentation"
by Long et al. We trained and tested ```FCN-32s```, ```FCN-16s``` and ```FCN-8s``` against PASCAL VOC 2012
dataset.

You can find all the scripts that were used for training and evaluation [here](dense_ai/recipes/pascal_voc/segmentation).

This code has been used to train networks with this performance:

| Model            | Test data |Mean IOU | Mean pix. accuracy | Pixel accuracy|Inference time (512x512 px. image) | Model Download Link |
|------------------|-----------|---------|--------------------|----------------|----|---------------------|
| FCN-32s (ours)   | RV-VOC12  | 60.0   | in prog.           | in prog.       |41 ms.| [Dropbox](https://www.dropbox.com/s/66coqapbva7jpnt/fcn_32s.tar.gz?dl=0)            |
| FCN-16s (ours)   | RV-VOC12  | in prog.   | in prog.           | in prog.  | in prog.     | [Dropbox](https://www.dropbox.com/s/tmhblqcwqvt2zjo/fcn_16s.tar.gz?dl=0)            |
| FCN-8s (ours)    | RV-VOC12  | in prog.   | in prog.           | in prog.   | in prog    | [Dropbox](https://www.dropbox.com/s/7r6lnilgt78ljia/fcn_8s.tar.gz?dl=0)            |
| FCN-32s (orig.)  | RV-VOC11  | 59.40   | 73.30              | 89.10         | in prog. |                     |
| FCN-16s (orig.)  | RV-VOC11  | 62.40   | 75.70              | 90.00         | in prog. |                     |
| FCN-8s  (orig.)  | RV-VOC11  | 62.70   | 75.90              | 90.30         | in prog. |                     |



## About

If you used the code for your research, please, cite the paper:

    @article{pakhomov2017deep,
      title={Deep Residual Learning for Instrument Segmentation in Robotic Surgery},
      author={Pakhomov, Daniil and Premachandran, Vittal and Allan, Max and Azizian, Mahdi and Navab, Nassir},
      journal={arXiv preprint arXiv:1703.08580},
      year={2017}
    }

During implementation, some preliminary experiments and notes were reported:
- [Converting Image Classification network into FCN](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/10/30/image-classification-and-segmentation-using-tensorflow-and-tf-slim/)
- [Performing upsampling using transposed convolution](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/)
- [Conditional Random Fields for Refining of Segmentation and Coarseness of FCN-32s model segmentations](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/18/image-segmentation-with-tensorflow-using-cnns-and-conditional-random-fields/)
- [TF-records usage](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/)