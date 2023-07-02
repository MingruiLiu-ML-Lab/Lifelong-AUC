#!/bin/bash

MY_PYTHON="python"

CIFAR_100i="--dataset CIFAR100 --n_epochs 1 --make_imbalanced yes --imratio 0.05 --lr 0.1 --n_memories 64 --data_path data/ --save_path results/
--batch_size 64 --log_every 100 --samples_per_task -1 --data_file cifar100.pt  --cuda yes --seed 0"

CUB200="--dataset CUB200 --data_path data/CUB_data/CUB_200_2011/images/ --make_imbalanced yes --imratio 0.05 --lr 0.1 --n_memories 64  --shuffle_tasks no --save_path results\
 --batch_size 64 --log_every 100 --samples_per_task -1  --cuda yes --seed 0"

AWA2="--dataset AWA2 --data_path data/AWA2 --make_imbalanced yes --imratio 0.05 --lr 0.1 --n_memories 64  --shuffle_tasks no --save_path results\
 --batch_size 64 --log_every 100 --samples_per_task -1  --cuda yes --seed 0"

ISIC_SPLIT="--n_epochs 1 --make_imbalanced yes --imratio 0.05 --lr 0.1 --n_memories 128 --data_path data --shuffle_tasks no --save_path results\
 --batch_size 128 --log_every 100 --samples_per_task -1 --data_file isic.pt  --cuda yes --seed 0"

EuroSat_SPLIT="--n_epochs 1 --make_imbalanced yes --imratio 0.05 --lr 0.1 --n_memories 128 --data_path data --shuffle_tasks no --save_path results\
 --batch_size 128 --log_every 100 --samples_per_task -1 --data_file eurosat_split.pt  --cuda yes --seed 0"


echo "------------start training-------------------"

$MY_PYTHON main.py $CIFAR_100i --model auc --batch_size
$MY_PYTHON main_loader.py $CUB200 --model auc --seed $i
$MY_PYTHON main_loader.py $AWA2 --model auc --seed $i


# hyper-parameter search
## ------------------------------------------------
#$MY_PYTHON main.py $CIFAR_100i --model auc --lr 0.1
#$MY_PYTHON main.py $CIFAR_100i --model auc --lr 0.3
#$MY_PYTHON main.py $CIFAR_100i --model auc --lr 0.03

#$MY_PYTHON main.py $CIFAR_100i --model auc --batch_size 32
#$MY_PYTHON main.py $CIFAR_100i --model auc --batch_size 64
#$MY_PYTHON main.py $CIFAR_100i --model auc --batch_size 128

#$MY_PYTHON main_loader.py $CUB200 --model auc --lr 0.1
#$MY_PYTHON main_loader.py $CUB200 --model auc --lr 0.3
#$MY_PYTHON main_loader.py $CUB200 --model auc --lr 0.03

#$MY_PYTHON main_loader.py $CUB200 --model auc --batch_size 32 --seed 1
#$MY_PYTHON main_loader.py $CUB200 --model auc --batch_size 64
#$MY_PYTHON main_loader.py $CUB200 --model auc --batch_size 128
## ------------------------------------------------



#for((i=0;i<5;i++));
#do
#$MY_PYTHON main.py $CIFAR_100i --model rm --seed $i --batch_size $memory
#$MY_PYTHON main.py $CIFAR_100i --model rm_ours --seed $i --batch_size $memory
#$MY_PYTHON main.py $CIFAR_100i --model CBRS_agem --seed $i
#$MY_PYTHON main.py $CIFAR_100i --model CBRS_mega --seed $i
#$MY_PYTHON main.py $CIFAR_100i --model CBRS_gdumb --seed $i
#$MY_PYTHON main.py $CIFAR_100i --model CBRS_ours --seed $i
#$MY_PYTHON main.py $CIFAR_100i --model auc --batch_size $memory --seed $i

#
#$MY_PYTHON main_loader.py $CUB200 --model CBRS_agem --seed $i
#$MY_PYTHON main_loader.py $CUB200 --model CBRS_mega --seed $i
#$MY_PYTHON main_loader.py $CUB200 --model CBRS_gdumb --seed $i
#$MY_PYTHON main_loader.py $CUB200 --model CBRS_ours --seed $i
#
#$MY_PYTHON main.py $AWA2 --model rm --seed $i
#$MY_PYTHON main_loader.py $AWA2 --model CBRS_agem --seed $i
#$MY_PYTHON main_loader.py $AWA2 --model CBRS_mega --seed $i
#$MY_PYTHON main_loader.py $AWA2 --model CBRS_gdumb --seed $i
#$MY_PYTHON main_loader.py $AWA2 --model CBRS_ours --seed $i
#done


#for((i=0;i<3;i++));
#do
#$MY_PYTHON main_loader.py $CUB200 --model auc_mem_balanced --seed $i
#done


#$MY_PYTHON main.py $CIFAR_100i --model mega
#$MY_PYTHON main.py $CIFAR_100i --model auc
#$MY_PYTHON main.py $CIFAR_100i --model mega_two_model

#for((i=0;i<1;i++));
#do
## $MY_PYTHON main.py $CIFAR_100i --model der_auc_loss --seed $i
## $MY_PYTHON main_loader.py $CUB200 --model der_auc_loss --seed $i
## $MY_PYTHON main_loader.py $AWA2 --model der_auc_loss  --seed $i
#
# $MY_PYTHON main.py $CIFAR_100i --model ewc_auc_loss --seed $i
# $MY_PYTHON main_loader.py $CUB200 --model ewc_auc_loss --seed $i
# $MY_PYTHON main_loader.py $AWA2 --model ewc_auc_loss  --seed $i
#done
# CIFAR

#for((i=0;i<1;i++));
#do
#$MY_PYTHON main.py $CIFAR_100i --model single --seed $i
#$MY_PYTHON main.py $CIFAR_100i --model ewc --seed $i
#$MY_PYTHON main.py $CIFAR_100i --model mas --seed $i
#$MY_PYTHON main.py $CIFAR_100i --model gem --seed $i
#$MY_PYTHON main.py $CIFAR_100i --model gdumb --seed $i
#$MY_PYTHON main.py $CIFAR_100i --model agem --seed $i
#$MY_PYTHON main.py $CIFAR_100i --model mega --seed $i
#$MY_PYTHON main.py $CIFAR_100i --model auc_one_model --seed $i
#$MY_PYTHON main.py $CIFAR_100i --model auc --seed $i
#done


# CUB200
#for((i=0;i<1;i++));
#do
#$MY_PYTHON main_loader.py $CUB200 --model single --seed $i
#$MY_PYTHON main_loader.py $CUB200 --model ewc --seed $i
#$MY_PYTHON main_loader.py $CUB200 --model mas --seed $i
#$MY_PYTHON main_loader.py $CUB200 --model gem --seed $i
#$MY_PYTHON main_loader.py $CUB200 --model agem --seed $i
#$MY_PYTHON main_loader.py $CUB200 --model mega  --seed $i
#$MY_PYTHON main_loader.py $CUB200 --model gdumb --seed $i
#$MY_PYTHON main_loader.py $CUB200 --model auc --seed $i
#$MY_PYTHON main_loader.py $CUB200 --model auc_mem --seed $i
#$MY_PYTHON main_loader.py $CUB200 --model auc_one_model # --seed $i
#done

# AWA2
#for((i=0;i<1;i++));
#do
#$MY_PYTHON main_loader.py $AWA2 --model single --seed $i
#$MY_PYTHON main_loader.py $AWA2 --model ewc --seed $i
#$MY_PYTHON main_loader.py $AWA2 --model mas --seed $i
#$MY_PYTHON main_loader.py $AWA2 --model gem --seed $i
#$MY_PYTHON main_loader.py $AWA2 --model agem --seed $i
#$MY_PYTHON main_loader.py $AWA2 --model mega --seed $i
#$MY_PYTHON main_loader.py $AWA2 --model auc --seed $i
#$MY_PYTHON main_loader.py $AWA2 --model auc_mem --seed $i
#$MY_PYTHON main_loader.py $AWA2 --model auc_one_model --seed $i
#done

# ISIC_SPLIT
#$MY_PYTHON main.py $ISIC_SPLIT --model single
#$MY_PYTHON main.py $ISIC_SPLIT --model ewc
#$MY_PYTHON main.py $ISIC_SPLIT --model mas
#$MY_PYTHON main.py $ISIC_SPLIT --model gem
#$MY_PYTHON main.py $ISIC_SPLIT --model agem
#$MY_PYTHON main.py $ISIC_SPLIT --model mega
##$MY_PYTHON main.py $ISIC --model mega_two_model
#$MY_PYTHON main.py $ISIC_SPLIT --model auc_one_model
#$MY_PYTHON main.py $ISIC_SPLIT --model auc


# ISIC_ROTATIONS
#$MY_PYTHON main.py $ISIC_ROTATIONS --model single
#$MY_PYTHON main.py $ISIC_ROTATIONS --model auc
#$MY_PYTHON main.py $ISIC_ROTATIONS --model ewc
#$MY_PYTHON main.py $ISIC_ROTATIONS --model mas
#$MY_PYTHON main.py $ISIC_ROTATIONS --model gem
#$MY_PYTHON main.py $ISIC_ROTATIONS --model agem
#$MY_PYTHON main.py $ISIC_ROTATIONS --model mega
##$MY_PYTHON main.py $ISIC_ROTATIONS --model mega_two_model
#$MY_PYTHON main.py $ISIC_ROTATIONS --model auc_one_model


# EUROSAT_ROTATIONS
#$MY_PYTHON main.py $EuroSat_ROTATIONS --model single
#$MY_PYTHON main.py $EuroSat_ROTATIONS --model ewc
#$MY_PYTHON main.py $EuroSat_ROTATIONS --model mas
#$MY_PYTHON main.py $EuroSat_ROTATIONS --model gem
#$MY_PYTHON main.py $EuroSat_ROTATIONS --model agem
#$MY_PYTHON main.py $EuroSat_ROTATIONS --model mega
#$MY_PYTHON main.py $EuroSat_ROTATIONS --model auc_one_modelF
#$MY_PYTHON main.py $EuroSat_ROTATIONS --model aucF

# EUROSAT_SPLIT
#$MY_PYTHON main.py $EuroSat_SPLIT --model single
#$MY_PYTHON main.py $EuroSat_SPLIT --model ewc
#$MY_PYTHON main.py $EuroSat_SPLIT --model mas
#$MY_PYTHON main.py $EuroSat_SPLIT --model gem
#$MY_PYTHON main.py $EuroSat_SPLIT --model agem
#$MY_PYTHON main.py $EuroSat_SPLIT --model mega
#$MY_PYTHON main.py $EuroSat_SPLIT --model auc_one_model
#$MY_PYTHON main.py $EuroSat_SPLIT --model auc
#
