#!/bin/bash
conda activate openmmlab;
python CLStrain.py --modelname 'ViT' --work_dir './work_dirs/ViT_full_uesat';
python CLStrain.py --modelname 'ViT' --data_list 'sam5_VAE3_kmeans.txt' --work_dir './work_dirs/ViT5_vae3_uesat';
python CLStrain.py --modelname 'ViT' --data_list 'sam10_VAE3_kmeans.txt' --work_dir './work_dirs/ViT10_vae3_uesat';
python CLStrain.py --modelname 'ViT' --data_list 'sam15_VAE3_kmeans.txt' --work_dir './work_dirs/ViT15_vae3_uesat';
python CLStrain.py --modelname 'ViT' --data_list 'sam20_VAE3_kmeans.txt' --work_dir './work_dirs/ViT20_vae3_uesat';
python CLStrain.py --modelname 'ViT' --data_list 'uesatL_Random5_MMdata.txt' --work_dir ./work_dirs/ViT5_random_uesat;
python CLStrain.py --modelname 'ViT' --data_list 'uesatL_Random10_MMdata.txt' --work_dir ./work_dirs/ViT10_random_uesat;
python CLStrain.py --modelname 'ViT' --data_list 'uesatL_Random15_MMdata.txt' --work_dir ./work_dirs/ViT15_random_uesat;
python CLStrain.py --modelname 'ViT' --data_list 'uesatL_Random20_MMdata.txt' --work_dir ./work_dirs/ViT20_random_uesat;