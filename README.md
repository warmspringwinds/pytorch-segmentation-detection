# Image Segmentation and Object Detection in Pytorch 

```Pytorch-Segmentation-Detection``` is a library for image segmentation and object detection with reported results achieved on common image segmentation/object detection datasets, pretrained models and scripts to reproduce them.


# Segmentation


## PASCAL VOC 2012

Implemented models were tested on Restricted PASCAL VOC 2012 Validation dataset (RV-VOC12) or Full PASCAL VOC 2012 Validation dataset (VOC-2012) and trained on
the PASCAL VOC 2012 Training data and additional Berkeley segmentation data for PASCAL VOC 12.

You can find all the scripts that were used for training and evaluation [here](pytorch_segmentation_detection/recipes/pascal_voc/segmentation).

This code has been used to train networks with this performance:

| Model            | Test data |Mean IOU | Mean pix. accuracy | Pixel accuracy|Inference time (512x512 px. image) | Model Download Link | Related paper |
|------------------|-----------|---------|--------------------|----------------|----|---------------------|----------|
| Resnet-18-8s    | RV-VOC12  | 59.0   | in prog.           | in prog.       |28 ms.| [Dropbox](https://www.dropbox.com/s/zxv1hb09fa8numa/resnet_18_8s_59.pth?dl=0)            | [DeepLab](https://arxiv.org/abs/1606.00915) |
| Resnet-34-8s   | RV-VOC12  | 68.0   | in prog.           | in prog.  | 50 ms.  | [Dropbox](https://www.dropbox.com/s/91wcu6bpqezu4br/resnet_34_8s_68.pth?dl=0)            | [DeepLab](https://arxiv.org/abs/1606.00915) |
| Resnet-50-16s   | VOC12  | 66.5   | in prog.           | in prog.  | in prog.  | in prog.        | [DeepLab](https://arxiv.org/abs/1606.00915) |
| Resnet-50-8s   | VOC12  | 67.0   | in prog.           | in prog.  | in prog.  | in prog.        | [DeepLab](https://arxiv.org/abs/1606.00915) |
| Resnet-50-8s-deep-sup   | VOC12  | 67.1   | in prog.           | in prog.  | in prog.  | in prog.        | [DeepLab](https://arxiv.org/abs/1606.00915) |
| Resnet-101-16s   | VOC12  | 68.6   | in prog.           | in prog.  | in prog.  | in prog.        | [DeepLab](https://arxiv.org/abs/1606.00915) |
| PSP-Resnet-18-8s  | VOC12  | 68.3   | n/a              | n/a         | n/a |     in prog.                | [PSPnet](https://arxiv.org/abs/1612.01105) |
| PSP-Resnet-50-8s  | VOC12  | 73.6   | n/a              | n/a         | n/a |     in prog.                | [PSPnet](https://arxiv.org/abs/1612.01105) |


Some qualitative results:

![Alt text](pytorch_segmentation_detection/recipes/pascal_voc/segmentation/segmentation_demo_preview.gif?raw=true "Title")


## Endovis 2017

Implemented models were trained on Endovis 2017 segmentation dataset and the sequence number
3 was used for validation and was not included in training dataset. 

The code to acquire the training and validating the model is also provided in the library.

Additional Qualitative results can be found on [this youtube playlist](https://www.youtube.com/watch?v=DJZxOuT5GY0&list=PLJkMX36nfYD3MpJozA3kdJKQpTVishk5_).

### Binary Segmentation

| Model            | Test data |Mean IOU | Mean pix. accuracy | Pixel accuracy|Inference time (512x512 px. image) | Model Download Link |
|------------------|-----------|---------|--------------------|----------------|----|---------------------|
| Resnet-9-8s   | Seq # 3 *  | 96.1   | in prog.           | in prog.       |13.3 ms.| [Dropbox](https://www.dropbox.com/s/3l7o1sfrnqhnpw8/resnet_9_8s.pth?dl=0)            |
| Resnet-18-8s   | Seq # 3  | 96.0   | in prog.           | in prog.       |28 ms.| [Dropbox](https://www.dropbox.com/s/4lemtiaacrytatu/resnet_18_8s_best.pth?dl=0)            |
| Resnet-34-8s   | Seq # 3  | in prog.   | in prog.           | in prog.  | 50 ms.  | in prog.            |

Resnet-9-8s network was tested on the 0.5 reduced resoulution (512 x 640).

Qualitative results (on validation sequence):

![Alt text](pytorch_segmentation_detection/recipes/endovis_2017/segmentation/validation_binary.gif?raw=true "Title")

### Multi-class Segmentation

| Model            | Test data |Mean IOU | Mean pix. accuracy | Pixel accuracy|Inference time (512x512 px. image) | Model Download Link |
|------------------|-----------|---------|--------------------|----------------|----|---------------------|
| Resnet-18-8s   | Seq # 3  | 81.0   | in prog.           | in prog.       |28 ms.| [Dropbox](https://www.dropbox.com/s/p9ey655mmzb3v5l/resnet_18_8s_multiclass_best.pth?dl=0)            |
| Resnet-34-8s   | Seq # 3  | in prog.   | in prog.           | in prog.  | 50 ms.  | in prog            |

Qualitative results (on validation sequence):

![Alt text](pytorch_segmentation_detection/recipes/endovis_2017/segmentation/validation_multiclass.gif?raw=true "Title")


## Cityscapes

 The dataset contains video sequences recorded in street scenes from 50 different cities, with high quality pixel-level annotations of  ```5 000``` frames. The annotations contain ```19``` classes which represent cars, road, traffic signs and so on.
 
 | Model            | Test data |Mean IOU | Mean pix. accuracy | Pixel accuracy|Inference time (512x512 px. image) | Model Download Link |
|------------------|-----------|---------|--------------------|----------------|----|---------------------|
| Resnet-18-32s  | Validation set  | 61.0   | in prog.           | in prog.  | in prog.  | in prog.           |
| Resnet-18-8s   | Validation set  | 60.0   | in prog.           | in prog.       |28 ms.| [Dropbox](https://www.dropbox.com/s/vdy4sqkk2s3f5v5/resnet_18_8s_cityscapes_best.pth?dl=0)            |
| Resnet-34-8s   | Validation set  | 69.1   | in prog.           | in prog.  | 50 ms.  | [Dropbox](https://www.dropbox.com/s/jeaw9ny0jtl60uc/resnet_34_8s_cityscapes_best.pth?dl=0)           |
| Resnet-50-16s-PSP   | Validation set  | 71.2   | in prog.           | in prog.  | in prog.  | in prog.           |

Qualitative results (on validation sequence):

Whole sequence can be viewed [here](https://www.youtube.com/watch?v=rYYbmYXmC0E).

![Alt text](pytorch_segmentation_detection/recipes/cityscapes/cityscapes_demo.gif?raw=true "Title")


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
