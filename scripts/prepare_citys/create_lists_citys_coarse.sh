#!/usr/bin/env bash

find ./leftImg8bit/train_extra -maxdepth 3 -name "*_leftImg8bit.png" | sort > train_images.txt
find ./leftImg8bit/val -maxdepth 3 -name "*_leftImg8bit.png" | sort > val_images.txt

find ./gtCoarse/train_extra -maxdepth 4 -name "*_trainIds.png" | sort > train_labels.txt
find ./gtCoarse/val -maxdepth 4 -name "*_trainIds.png" | sort > val_labels.txt
