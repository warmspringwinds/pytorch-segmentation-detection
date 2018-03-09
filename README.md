# Image Segmentation and Object Detection in Pytorch 

```Pytorch-Segmentation-Detection``` is a library for dense inference and training of Convolutional Neural Networks (CNNs) on Images for Segmentation and Detection.
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

2. Some libraries which can be acquired by installing [Anaconda package](https://www.continuum.io/downloads).
 
    Or you can install ```scikit-image```, ```matplotlib```, ```numpy``` using ```pip```.
 
3. Clone the library:

 ```git clone --recursive https://github.com/warmspringwinds/pytorch-segmentation-detection```
 
   And use this code snippet before you start to use the library:
 
   ```python
   import sys
   # update with your path
   # All the jupyter notebooks in the repository already have this
   sys.path.append("/your/path/pytorch-segmentation-detection/")
   sys.path.insert(0, '/your/path/pytorch-segmentation-detection/vision/')
   ```
   Here we use our [pytorch/vision](https://github.com/pytorch/vision) fork, which might
   be [merged](https://github.com/pytorch/vision/pull/184) and [futher merged](https://github.com/pytorch/vision/pull/190) in a future.
   We have added it as a submodule to our repository.

4. Download segmentation or detection models that you want to use manually (links can be found below).

## PASCAL VOC 2012 (Segmentation)

Implemented models were tested on Restricted PASCAL VOC 2012 Validation dataset (RV-VOC12) and trained on
the PASCAL VOC 2012 Training data and additional Berkeley segmentation data for PASCAL VOC 12.
It was important to test models on restricted Validation dataset to make sure no images in the
validation dataset were seen by model during training.

The code to acquire the training and validating the model is also provided in the library.


### DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs

Here you can find models that were described in the paper "DeepLab: Semantic Image Segmentation with Deep 
Convolutional Nets, Atrous Convolution, and Fully Connected CRFs" by Chen et al. We trained and tested 
```Resnet-18-8s```, ```Resnet-34-8s``` against PASCAL VOC 2012 dataset.

You can find all the scripts that were used for training and evaluation [here](pytorch_segmentation_detection/recipes/pascal_voc/segmentation).

Qualitative results:

![Alt text](pytorch_segmentation_detection/recipes/pascal_voc/segmentation/segmentation_demo_preview.gif?raw=true "Title")

This code has been used to train networks with this performance:

| Model            | Test data |Mean IOU | Mean pix. accuracy | Pixel accuracy|Inference time (512x512 px. image) | Model Download Link |
|------------------|-----------|---------|--------------------|----------------|----|---------------------|
| Resnet-18-8s (ours)   | RV-VOC12  | 59.0   | in prog.           | in prog.       |28 ms.| [Dropbox](https://www.dropbox.com/s/zxv1hb09fa8numa/resnet_18_8s_59.pth?dl=0)            |
| Resnet-34-8s (ours)   | RV-VOC12  | 68.0   | in prog.           | in prog.  | 50 ms.  | [Dropbox](https://www.dropbox.com/s/91wcu6bpqezu4br/resnet_34_8s_68.pth?dl=0)            |
| Resnet-50-8s (ours)    | RV-VOC12  | in prog.   | in prog.           | in prog.   | in prog    | in prog.            |
| Resnet-101-8s (ours)    | RV-VOC12  | in prog.   | in prog.           | in prog.   | in prog    | in prog.            |
| Resnet-101-16s (orig)  | RV-VOC11  | 69.0   | n/a              | n/a         | 180 ms. |                     |


### Fully Convolutional Networks for Semantic Segmentation (FCNs)

Here you can find models that were described in the paper "Fully Convolutional Networks for Semantic Segmentation"
by Long et al. We trained and tested ```FCN-32s```, ```FCN-16s``` (in prog.) and ```FCN-8s``` (in prog.) against PASCAL VOC 2012
dataset.

You can find all the scripts that were used for training and evaluation [here](pytorch_segmentation_detection/recipes/pascal_voc/segmentation).

This code has been used to train networks with this performance:

| Model            | Test data |Mean IOU | Mean pix. accuracy | Pixel accuracy|Inference time (512x512 px. image) | Model Download Link |
|------------------|-----------|---------|--------------------|----------------|----|---------------------|
| FCN-32s (ours)   | RV-VOC12  | 60.0   | in prog.           | in prog.       |41 ms.| [Dropbox](https://www.dropbox.com/s/8l049d19k46ts9b/fcn_32s_best.pth?dl=0)            |
| FCN-16s (ours)   | RV-VOC12  | in prog.   | in prog.           | in prog.  | in prog.     | in prog.            |
| FCN-8s (ours)    | RV-VOC12  | in prog.   | in prog.           | in prog.   | in prog    | in prog.            |
| FCN-32s (orig.)  | RV-VOC11  | 59.40   | 73.30              | 89.10         | in prog. |                     |
| FCN-16s (orig.)  | RV-VOC11  | 62.40   | 75.70              | 90.00         | in prog. |                     |
| FCN-8s  (orig.)  | RV-VOC11  | 62.70   | 75.90              | 90.30         | in prog. |                     |


## Endovis 2017 (Segmentation)

Implemented models were trained on Endovis 2017 segmentation dataset and the sequence number
3 was used for validation and was not included in training dataset. 

The code to acquire the training and validating the model is also provided in the library.

Additional Qualitative results can be found on [this youtube playlist](https://www.youtube.com/watch?v=DJZxOuT5GY0&list=PLJkMX36nfYD3MpJozA3kdJKQpTVishk5_).

### Binary Segmentation

| Model            | Test data |Mean IOU | Mean pix. accuracy | Pixel accuracy|Inference time (512x512 px. image) | Model Download Link |
|------------------|-----------|---------|--------------------|----------------|----|---------------------|
| Resnet-18-8s   | RV-VOC12  | 96.0   | in prog.           | in prog.       |28 ms.| [Dropbox](https://www.dropbox.com/s/4lemtiaacrytatu/resnet_18_8s_best.pth?dl=0)            |
| Resnet-34-8s   | RV-VOC12  | in prog.   | in prog.           | in prog.  | 50 ms.  | in prog.            |


Qualitative results (on validation sequence):

![Alt text](pytorch_segmentation_detection/recipes/endovis_2017/segmentation/validation_binary.gif?raw=true "Title")

### Multi-class Segmentation

| Model            | Test data |Mean IOU | Mean pix. accuracy | Pixel accuracy|Inference time (512x512 px. image) | Model Download Link |
|------------------|-----------|---------|--------------------|----------------|----|---------------------|
| Resnet-18-8s   | RV-VOC12  | 81.0   | in prog.           | in prog.       |28 ms.| [Dropbox](https://www.dropbox.com/s/p9ey655mmzb3v5l/resnet_18_8s_multiclass_best.pth?dl=0)            |
| Resnet-34-8s   | RV-VOC12  | in prog.   | in prog.           | in prog.  | 50 ms.  | in prog            |

Qualitative results (on validation sequence):

![Alt text](pytorch_segmentation_detection/recipes/endovis_2017/segmentation/validation_multiclass.gif?raw=true "Title")

## Applications

We demonstrate applications of our library for a certain tasks which are being ported/ has already been ported to mobile devices:

1. [Sticker creation](pytorch_segmentation_detection/recipes/pascal_voc/segmentation/resnet_34_8s_demo.ipynb)

2. [Iphone's portait effect](pytorch_segmentation_detection/recipes/pascal_voc/segmentation/resnet_34_8s_demo.ipynb)

3. [Background replacement](pytorch_segmentation_detection/recipes/pascal_voc/segmentation/resnet_34_8s_demo.ipynb)

4. Surgical Robotic Tools Segmentation (see below)

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
