#!/bin/bash

export PATH=/home/siddiqui/anaconda3/bin/:$PATH

# sudo userdocker run -it -v /netscratch:/netscratch dlcc/tensorflow_opencv /netscratch/siddiqui/Repositories/Segmentation-PyTorch/train.sh
cd /netscratch/siddiqui/Repositories/Segmentation-PyTorch/

echo "Training FCN on ICDAR-13 Structure Recognition"
# python train_str.py --dataset table_str
python train_str.py --dataset table_str_two_heads --img_rows 256 --img_cols 2048