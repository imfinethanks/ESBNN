#!/bin/bash
clear
cd /mnt/vepfs/ML/ml-users/xulong/code_dir_bnn/BNext-main/src
source activate py39
python3 train_assistant_group_amp.py --model bnext18 --teacher_num 0 --assistant_teacher_num 0  --workers=4 --mixup 0.0 --cutmix 0.0 --aug-repeats 1 --dali_cpu  --multiprocessing-distributed --dist-url 'tcp://127.0.0.1:33499' --dist-backend 'nccl' --world-size 1 --rank 0  --dataset=cifar10   --batch_size 512 --learning_rate=1e-3  --epochs=512 --weight_decay=0 | tee -a log/training.txt


