###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2018
###########################################################################

import os
import sys
import random
import numpy as np
from tqdm import tqdm, trange
from PIL import Image, ImageOps, ImageFilter

import torch
import torch.utils.data as data
import torchvision.transforms as transform

from .base import BaseDataset

CITYSCAPE_PALETTE = np.asarray([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0,0,0]
    ], dtype=np.uint8)

LabelID = np.asarray([7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33,0],dtype=np.uint8)


TRIPLET_PALETTE = np.asarray([
    [0, 0, 0, 255],
    [217, 83, 79, 255],
    [91, 192, 222, 255]], dtype=np.uint8)

class CityCoarseSegmentation(BaseDataset):
    NUM_CLASS = 19
    def __init__(self, root=os.path.expanduser('/data2/.encoding/data'), split='train',
                 mode=None, transform=None, list_dir=None,target_transform=None, **kwargs):
        super(CityCoarseSegmentation, self).__init__(
            root, split, mode, transform, target_transform, **kwargs)

        self.data_dir = os.path.join(root,'cityscapes_coarse')

        self.list_dir = self.data_dir if list_dir is None else list_dir

        self.read_lists()

        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of: \
                " + self.root + "\n")


    def read_lists(self):
        image_path = os.path.join(self.list_dir, self.split + '_images.txt')
        label_path = os.path.join(self.list_dir, self.split + '_labels.txt')
        assert os.path.exists(image_path)
        self.images = [line.strip() for line in open(image_path, 'r')]
        if os.path.exists(label_path):
            self.mask_paths = [line.strip() for line in open(label_path, 'r')]
            assert len(self.images) == len(self.mask_paths)


    def __getitem__(self, index):
        tmp_path = self.images[index]
        tmp_path = os.path.join(self.data_dir,tmp_path)
        img = Image.open(tmp_path).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])

        tmp_path = self.mask_paths[index]
        tmp_path = os.path.join(self.data_dir, tmp_path)
        mask = Image.open(tmp_path)
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            mask = self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge from 480 to 720)
        short_size = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # random rotate -10~10, mask using NN rotate
        deg = random.uniform(-10, 10)
        img = img.rotate(deg, resample=Image.BILINEAR)
        mask = mask.rotate(deg, resample=Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        return img, self._mask_transform(mask)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        target = torch.from_numpy(target).long()
        # target = self._class_to_index(np.array(mask).astype('int32'))
        #trans = transform.ToTensor()
        #target = trans(mask)
        return target

    def __len__(self):
        return len(self.images)

    def make_pred(self, mask):
        # return lable images
        # return LabelID[predictions[ind]].astype(np.uint8)

        # return color images
        return CITYSCAPE_PALETTE[mask.squeeze()]


