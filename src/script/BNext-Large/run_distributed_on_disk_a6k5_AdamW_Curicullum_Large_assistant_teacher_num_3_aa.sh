#!/bin/bash
clear
cd /mnt/vepfs/ML/ml-users/xulong/code_dir_bnn/BNext-main/src
source activate py39
python3 train_assistant_group_amp.py --model bnext_large --distillation False --teacher_num 4 --assistant_teacher_num 3 --weak_teacher EfficientNet_B0 --mixup 0.0 --cutmix 0.0 --aug-repeats 1  --dali_cpu  --multiprocessing-distributed --dist-url 'tcp://127.0.0.1:33483' --dist-backend 'nccl' --world-size 1 --rank 0 --data=/mnt/vepfs/ML/ml-public/PublicDataset/BNN/ImageNet/  --batch_size 128 --learning_rate=1e-3  --epochs=512 --weight_decay=0 | tee -a log/training.txt


