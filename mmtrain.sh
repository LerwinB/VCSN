#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0;
# source ~/anaconda3/bin/activate openmmlab;

for i in 10 15 20 30 50 70;do
    python MMtrain.py configs/deeplabv3plus/deeplabv3plus_r50_SAM5_uesatrgb.py --cfg-options train_dataloader.dataset.split="sam${i}_VAE3n_kmeans.txt" --work-dir "work_dirs/deeplab_samVAE3_new${i}";
done

for per in 30 50 70; do
    #python MMtrain.py configs/deeplabv3plus/deeplabv3plus_r50_SAM5_uesatrgb.py --cfg-options train_dataloader.dataset.split="sam${per}_VAE3n_kmeans.txt" --work-dir "work_dirs/deeplabv3plus_r50_samvae${per}" --amp;
    python MMtrain.py configs/deeplabv3plus/deeplabv3plus_r50_SAM5_uesatrgb.py --cfg-options train_dataloader.dataset.split="uesat_Random${per}.txt" --work-dir "work_dirs/deeplabv3plus_r50_random${per}" --amp;
done


# python MMtrain.py configs/deeplabv3plus/deeplabv3plus_r50_SAM5_uesatrgb.py --cfg-options train_dataloader.dataset.split="sam5_UnetVAE_kmeans.txt" --work-dir "work_dirs/deeplabv3plus_r50_sam5_UnetVAE" --amp;
# python sam_active_select.py --task_name uesatL_SAM_vae3_full --data_pre True
# for per in 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95; do
#     python MMtrain.py configs/deeplabv3plus/deeplabv3plus_r50_SAM5_uesatrgb.py --cfg-options train_dataloader.dataset.split="uesat_Random${per}.txt" --work-dir "work_dirs/deeplabv3plus_r50_random${per}" --amp;
# done