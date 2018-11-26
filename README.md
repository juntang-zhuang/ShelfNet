# ShelfNet 
* [Link to my homepage](https://juntangzhuang.com)
* This is repository for my paper [ShelfNet for real-time semantic segmentation](https://juntang-zhuang.github.io/files/ShelfNet_2019.pdf), and avhieves both faster inference speed and higher segmentation accuracy, compared with other real-time models such as Lightweight-RefineNet.
* This repository only deals with Pascal dataset. Code for experiments on Cityscapes dataset can be found [here](https://github.com/juntang-zhuang/ShelfNet-Cityscapes).
* We tested ShelfNet with ResNet50 and ResNet101 as the backbone respectively: they achieved **59 FPS** and **42 FPS** respectively on a GTX 1080Ti GPU with a 512x512 input image. ShelfNet achieved high accuracy: on PASCAL VOC 2012 test set, it achieved **84.2%** mIoU with ResNet101 backbone and **82.8%** mIoU with ResNet50 backbone; it achieved **75.8%** mIoU with ResNet50 backbone on Cityscapes dataset.
* This implementation is based on [torch-encoding](https://github.com/zhanghang1989/PyTorch-Encoding). Main difference is the structure of the model.

# Requirements
* Please refer to [torch-encoding](https://github.com/zhanghang1989/PyTorch-Encoding) for implementation on synchronized batch-normalization layer.
* PyTorch 0.4.1
* requests
* nose
* scipy
* tqdm
* Other requirements by [torch-encoding](https://github.com/zhanghang1989/PyTorch-Encoding).

# How to run
* run ```python setup.py install``` to install torch-encoding
* make sure you have the same path for a datset in ```/scripts/prepare_xx.py``` and ```/encoding/datasets/xxx.py```
* run ```cd scripts```
* run ```python prepared_xx.py ``` to prepare datasets, including MS COCO, PASCAL VOC, PASCAL Aug, PASCAL Context

# Examples on Pascal VOC datasets
![Pascal results](https://github.com/juntang-zhuang/ShelfNet/blob/master/video_demo/Pascal_results.png) </br>

# Video Demo on Cityscapes datasets
**Video demo of ShelfNet50 on Cityscapes**
<a href="url"><img src="https://github.com/juntang-zhuang/ShelfNet/blob/master/video_demo/shelfnet50_demo.gif" align="left"  width="1000" ></a> </br>
**Video demo of ShelfNet101 on Cityscapes** </br>
<a href="url"><img src="https://github.com/juntang-zhuang/ShelfNet/blob/master/video_demo/shelfnet101_demo.gif" align="left"  width="1000" ></a> </br>
