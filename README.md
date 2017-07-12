# Pytorch-image-segmentation: Image Segmentation framework

The aim of the ```TF Image Segmentation``` framework is to provide/provide a simplified way for:

- Converting some popular general/medical/other Image Segmentation Datasets into easy-to-use for training ```.tfrecords```
format with unified interface: different datasets but same way to store images and annotations.
- Training routine with on-the-fly data augmentation (scaling, color distortion).
- Training routine that is proved to work for particular model/dataset pair.
- Evaluating Accuracy of trained models with common accuracy measures: Mean IOU, Mean pix. accuracy, Pixel accuracy.
- Model files that were trained on a particular dataset with reported accuracy (models that were trained using
TF with reported training routine and not models that were converted from Caffe or other framework)
- Model definitions (like FCN-32s and others) that use weights initializations from Image Classification models like
VGG that are officially provided by TF-Slim library.

So far, the framework contains an implementation of the FCN models (training
and evaluation) in Tensorflow and TF-Slim library with training routine, reported accuracy,
trained models for PASCAL VOC 2012 dataset. To train these models on your data, [convert your dataset
to tfrecords](tf_image_segmentation/recipes/pascal_voc/convert_pascal_voc_to_tfrecords.ipynb) and follow the
instructions below.

The end goal is to provide utilities to convert other datasets, report accuracies on them and provide models.

## Installation

This code requires:

1. Tensorflow ```r0.12``` or later version.

2. Custom [tensorflow/models](https://github.com/tensorflow/models) repository, which might be [merged](https://github.com/tensorflow/models/pull/684) in a future.

 Simply run:
 
 ```git clone -b fully_conv_vgg https://github.com/warmspringwinds/models```
 
 And add ```models/slim``` subdirectory to your path:

 ```python
 import sys
 # update with your path
 sys.path.append('/home/dpakhom1/workspace/models/slim/')
 ```
3. Some libraries which can be acquired by installing [Anaconda package](https://www.continuum.io/downloads).
 
 Or you can install ```scikit-image```, ```matplotlib```, ```numpy``` using ```pip```.
 
4. ```VGG 16``` checkpoint file, which you can get from [here](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz).

5. Clone this library:

 ```git clone https://github.com/warmspringwinds/tf-image-segmentation```
 
 And add it to the path:
 
 ```python
 import sys
 # update with your path
 sys.path.append("/home/dpakhom1/tf_projects/segmentation/tf-image-segmentation/")
 ```


## PASCAL VOC 2012

Implemented models were tested on Restricted PASCAL VOC 2012 Validation dataset (RV-VOC12) and trained on
the PASCAL VOC 2012 Training data and additional Berkeley segmentation data for PASCAL VOC 12.
It was important to test models on restricted Validation dataset to make sure no images in the
validation dataset were seen by model during training.

The code to acquire the training and validating the model is also provided in the framework.

### Fully Convolutional Networks for Semantic Segmentation (FCNs)

Here you can find models that were described in the paper "Fully Convolutional Networks for Semantic Segmentation"
by Long et al. We trained and tested ```FCN-32s```, ```FCN-16s``` and ```FCN-8s``` against PASCAL VOC 2012
dataset.

You can find all the scripts that were used for training and evaluation [here](tf_image_segmentation/recipes/pascal_voc/FCNs).

This code has been used to train networks with this performance:

| Model            | Test data |Mean IOU | Mean pix. accuracy | Pixel accuracy | Model Download Link |
|------------------|-----------|---------|--------------------|----------------|---------------------|
| FCN-32s (ours)   | RV-VOC12  | 62.70   | in prog.           | in prog.       | [Dropbox](https://www.dropbox.com/s/66coqapbva7jpnt/fcn_32s.tar.gz?dl=0)            |
| FCN-16s (ours)   | RV-VOC12  | 63.52   | in prog.           | in prog.       | [Dropbox](https://www.dropbox.com/s/tmhblqcwqvt2zjo/fcn_16s.tar.gz?dl=0)            |
| FCN-8s (ours)    | RV-VOC12  | 63.65   | in prog.           | in prog.       | [Dropbox](https://www.dropbox.com/s/7r6lnilgt78ljia/fcn_8s.tar.gz?dl=0)            |
| FCN-32s (orig.)  | RV-VOC11  | 59.40   | 73.30              | 89.10          |                     |
| FCN-16s (orig.)  | RV-VOC11  | 62.40   | 75.70              | 90.00          |                     |
| FCN-8s  (orig.)  | RV-VOC11  | 62.70   | 75.90              | 90.30          |                     |



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