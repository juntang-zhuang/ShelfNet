from .base import *
#from .ade20k import ADE20KSegmentation
from .pascal_voc import VOCSegmentation
from .pascal_aug import VOCAugSegmentation
from .pcontext import ContextSegmentation
from .coco import COCOSegmentation
from .cityscapes import CitySegmentation
from .cityscapes_coarse import  CityCoarseSegmentation
datasets = {
    #'ade20k': ADE20KSegmentation,
    'pascal_voc': VOCSegmentation,
    'pascal_aug': VOCAugSegmentation,
    'pcontext': ContextSegmentation,
    'coco':COCOSegmentation,
    'citys': CitySegmentation,
    'citys_coarse':CityCoarseSegmentation,
}

def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
