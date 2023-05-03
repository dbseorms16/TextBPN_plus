#!/bin/bash

python -m torch.distributed.launch --nproc_per_node=7 train_textBPN.py --iter 5 --num_poly 16 --exp_name Custom --net resnet50 --scale 1 --max_epoch 660 --batch_size 40 --mgpu --gpu 0,1,2,3,4,5,6 --input_size 640 --optim Adam --lr 0.001 --num_workers 64 --viz --viz_freq 80

#--resume model/Ctw1500/TextBPN_resnet50_390.pth --start_epoch 665
#--resume model/Synthtext/TextBPN_resnet50_0.pth 
#--viz --viz_freq 80
#--start_epoch 300

###### test eval ############
# for ((i=10; i<=660; i=i+5))
#  do python3 eval_textBPN.py --exp_name Ctw1500 --checkepoch $i --test_size 640 1024 --dis_threshold 0.35 --cls_threshold 0.825 --gpu 0 --viz; 
# done

# python eval_textBPN.py --exp_name Custom --resume model/Ctw1500/TextBPN_resnet50_150.pth --test_size 640 1024 --dis_threshold 0.35 --cls_threshold 0.825 --gpu 0 --viz;




