#!/bin/bash

export PATH=/home/siddiqui/anaconda3/bin/:$PATH

# sudo userdocker run -it -v /netscratch:/netscratch dlcc/tensorflow_opencv /netscratch/siddiqui/Repositories/Segmentation-PyTorch/train.sh
cd /netscratch/siddiqui/Repositories/Segmentation-PyTorch/

echo "Training FCN on ICDAR-13 Structure Recognition"
# python train_str.py --dataset table_str
python train_str.py --dataset table_str_two_heads --img_rows 256 --img_cols 2048 --l_rate 1e-4 --n_epoch 250

# python test_str.py --img_path /netscratch/siddiqui/TableDetection/icdar_str_devkit/data/Images_original/eu-001-table-1-str.jpg --out_path ./out/ --dcrf --dataset table_str_two_heads --model_path ./segnet_two_heads_table_str_two_heads_best_model.pkl