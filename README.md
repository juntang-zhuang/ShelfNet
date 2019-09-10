# Introduction
* This is repository for paper [ShelfNet for fast semantic segmentation](https://arxiv.org/abs/1811.11254), implementing a fast segmentation CNN, and achieves both faster inference speed and higher segmentation accuracy, compared with other real-time models such as Lightweight-RefineNet.
* This branch performs experiments on Cityscapes dataset, please see branch ```pascal``` for experiments on PASCAL VOC dataset.
* Light-weight ShelfNet is implemented in repo [ShelfNet-lw-cityscapes](https://github.com/juntang-zhuang/ShelfNet-lw-cityscapes), it achieves 74.8% mIoU on Cityscapes in real-time tasks, with a speed of 59.2 FPS (61.7 FPS for BiSeNet at 74.7% on a GTX 1080Ti GPU); and achieves 79.0% mIoU in non real-time tasks with ResNet34 backbone (suparssing PSPNet and BiSeNet with ResNet50 and ResNet101 backbone).
* This implementation is based on [torch-encoding](https://github.com/zhanghang1989/PyTorch-Encoding). Main difference is the structure of the model. 

**Results**
* We tested ShelfNet with ResNet50 and ResNet101 as the backbone respectively: they achieved **59 FPS** and **42 FPS** respectively on a GTX 1080Ti GPU with a 512x512 input image. 
* On PASCAL VOC 2012 test set, it achieved **84.2%** mIoU with ResNet101 backbone and **82.8%** mIoU with ResNet50 backbone.
* It achieved **75.8%** mIoU with ResNet50 backbone on Cityscapes dataset.

**Differences from results reported in the paper on Cityscapes**
* Repo [ShelfNet-lw-cityscapes](https://github.com/juntang-zhuang/ShelfNet-lw-cityscapes) on Lightweight ShelfNet is exactly the same as reported in the paper. This repo for ShelfNet is slightly different.
* The results on PASCAL VOC is the same as in paper, but implementation on Cityscapes is slightly different.
* The result of ShelfNet50 is slightly different on this implementation and reported in the paper (75.4% in this implementation, 75.8% in the paper).
* The paper trains 500 epochs, while here the training epoch is 240.
* The paper does not use synchronized batch normalization, while this implementation uses synchronized batch normalization across multiple GPUs.
* For training on coarse labelled data, in this implementation the learning rate is set as 0.01 and remains constant; in results for the paper, the training on coarse labelled data uses a poly decay schedule, but the total epochs is set as 500, while I stopped the training mannualy at epoch 35 (In this way, there is a very slight decay on learning rate instead of constant).

# Requirements
* Please refer to [torch-encoding](https://github.com/zhanghang1989/PyTorch-Encoding) for implementation on synchronized batch-normalization layer.
* PyTorch 0.4.1 
* Python 3.6
* requests
* nose
* scipy
* tqdm
* Other requirements by [torch-encoding](https://github.com/zhanghang1989/PyTorch-Encoding).

# How to run
**Environment setup**
* run ```python setup.py install``` to install torch-encoding
* make sure you have the same path for a datset in ```/scripts/prepare_xx.py``` and ```/encoding/datasets/xxx.py```, default path is ```~/.encoding/data```, which is a hidden folder. You will need to type ```Ctrl + h``` to show is in ```Files```

**PASCAL dataset preparation**
* run ```cd scripts```
* run ```python prepared_xx.py ``` to prepare datasets, including MS COCO, PASCAL VOC, PASCAL Aug, PASCAL Context 
* Download test dataset from official evaluation server for PASCAL, extract and merge with training data folder, e.g. ```~/.encoding/data/VOCdevkit``` </br>

**Cityscapes dataset preparation**
* The data preparation code is modified from [fyu implementation](https://github.com/fyu/drn/tree/master/datasets/cityscapes)
* The scripts are in folder scripts/prepare_citys
* Step 1, download Cityscapes and Cityscapes Coarse dataset from [Cityscapes official website](https://www.cityscapes-dataset.com/downloads/), you need to download ```gtFine_trainvaltest.zip, gtFine_trainvaltest.zip , leftImg8bit_trainvaltest.zip, leftImg8bit_trainvaltest.zip ``` and unzip them into one folder 
* Step 2, prepare fine labelled dataset:</br>
  * convert original segmentation id into 19 training ids ```python3 scripts/prepare_citys/prepare_data.py <cityscape folder>/gtFine/```</br>
  * Run ```sh create_lists_citys.sh``` in cityscape data folder, and move ```info.json``` into the data folder
* Step 3, prepare coarse labelled dataset: <br>
  * convert original segmentation id into 19 training ids ```python3 scripts/prepare_citys/prepare_data.py <cityscape folder>/gtCoarse/```</br>
  * Run ```sh create_lists_citys_coarse.sh``` in cityscape data folder, and move ```info.json``` into the data folder
  
**Configurations** (refer to /experiments/option.py)</br>
* --diflr: default value is True. If set as True, the head uses 10x larger learning rate than the backbone; otherwise head and backbone uses the same learning rate.
* --model: which model to use, default is ```shelfnet```, other options include ```pspnet```, ```encnet```,```fcn```
* --backbone: backbone of the model, ```resnet50``` or ```resnet101```
* --dataset: which dataset to train on, ```coco``` for MS COCO, ```pascal_aug``` for augmented PASCAL,```pascal_voc``` for PASCAL VOC,```pcontext``` for pascal context.
* --aux: if type ```--aux```, the model will use auxilliray layer, which is a FCN head based on the final block of backbone.
* --se_loss: a context module based on final block of backbone, the shape is 1xm where m is number of categories. It penalizes whether a category is present or not.
* --resume: default is None. It specifies the checkpoint to load
* --ft: fine tune flag. If set as True, the code will resume from checkpoint but forget optimizer information.
* --checkname: folder name to store trained weights
* Other parameters are trevial, please refer to /experiments/segmentation/option.py for more details

**Training scripts on PASCAL VOC**
* run ```cd /experiments/segmentation```
* pre-train ShelfNet50 on COCO, </br>
```python train.py --backbone resnet50 --dataset coco --aux --se-loss --checkname ShelfNet50_aux```
* fine-tune ShelfNet50 on PASCAL_aug, you may need to double check the path for resume.</br>
```python train.py --backbone resnet50 --dataset pascal_aug --aux --se-loss --checkname ShelfNet50_aux --resume ./runs/coco/shelfnet/ShelfNet50_aux_se/model_best.pth.tar -ft```
* fine-tune ShelfNet50 on PASCAL VOC, you may need to double check the path for resume.</br>
```python train.py --backbone resnet50 --dataset pascal_voc --aux --se-loss --checkname ShelfNet50_aux --resume ./runs/pascal_aug/shelfnet/ShelfNet50_aux_se/model_best.pth.tar -ft```

**Training scripts on Cityscapes**
* run ```cd /experiments/segmentation```
* pre-train ShelfNet50 on coarse labelled dataset, </br>
```python train.py --diflr False --backbone resnet50 --dataset citys_coarse --checkname ShelfNet50_citys_coarse --lr-schedule step```
* fine-tune ShelfNet50 on fine labelled dataset, you may need to double check the path for resume.</br>
```python train.py --diflr False --backbone resnet50 --dataset citys --checkname citys_coarse --resume ./runs/citys_coarse/shelfnet/ShelfNet50_citys_coarse/model_best.pth.tar -ft```


**Test scripts on PASCAL VOC**
* To test on PASCAL_VOC with multiple-scales input \[0.5, 0.75, 1.0, 1.25, 1.5, 1.75\].</br>
```python test.py --backbone resnet50 --dataset pascal_voc --resume ./runs/pascal_voc/shelfnet/ShelfNet50_aux_se/model_best.pth.tar```
* To test on PASCAL_VOC with single-scale input</br>
```python test_single_scale.py --backbone resnet50 --dataset pascal_voc --resume ./runs/pascal_voc/shelfnet/ShelfNet50_aux_se/model_best.pth.tar```
* Similar experiments can be performed on ShelfNet with ResNet101 backbone, and experiments on Cityscapes can be performed by changing dataset as ```--dataset citys```

**Evaluation scripts**
* You can use the following script to generate ground truth - prediction pairs on PASCAL VOC validation set. </br>
```python evaluate_and_save.py --backbone resnet50 --dataset pascal_voc --resume ./runs/pascal_voc/shelfnet/ShelfNet50_aux_se/model_best.pth.tar --eval```

**Measure running speed**
* Measure running speed of ShelfNet on 512x512 image. </br>
```python test_speed.py --model shelfnet --backbone resnet101```</br>
```python test_speed.py --model pspnet --backbone resnet101```</br>

**Pre-trained weights**
* [Link to weights trained on PASCAL](https://drive.google.com/drive/folders/1k23TpBDsP9_gnb3LZlEcYyF4yoVzW99Z?usp=sharing)
* [Link to weights trained on Cityscapes](https://drive.google.com/drive/folders/1xwqJvmOw4BxiHrch8eWrBjUO_PyIBzQ-?usp=sharing)

# Structure of ShelfNet
![structure](video_demo/shelfnet.png) </br>


# Examples on Pascal VOC datasets
![Pascal results](video_demo/Pascal_results.png) </br>
# Video Demo on Cityscapes datasets
* Video demo of ShelfNet50 on Cityscapes
![Video demo of ShelfNet50](video_demo/shelfnet50_demo.gif) 
* Video demo of ShelfNet101 on Cityscapes
![Video demo of ShelfNet101](video_demo/shelfnet101_demo.gif)
