#!/bin/bash
python DETtrain.py det_configs/yolo/yolov3_d53_8xb8-ms-608-273e_uesat.py --cfg-option train_dataloader.dataset.split='uesatL_Random5_MMdata.txt' --work-dir work_dirs_det/yolov3-e20_Random5_uesat --amp
python DETtrain.py det_configs/yolo/yolov3_d53_8xb8-ms-608-273e_uesat.py --cfg-option train_dataloader.dataset.split='uesatL_Random10_MMdata.txt' --work-dir work_dirs_det/yolov3-e20_Random10_uesat --amp
python DETtrain.py det_configs/yolo/yolov3_d53_8xb8-ms-608-273e_uesat.py --cfg-option train_dataloader.dataset.split='uesatL_Random15_MMdata.txt' --work-dir work_dirs_det/yolov3-e20_Random15_uesat --amp
python DETtrain.py det_configs/yolo/yolov3_d53_8xb8-ms-608-273e_uesat.py --cfg-option train_dataloader.dataset.split='uesatL_Random20_MMdata.txt' --work-dir work_dirs_det/yolov3-e20_Random20_uesat --amp
python DETtrain.py det_configs/yolo/yolov3_d53_8xb8-ms-608-273e_uesat.py --cfg-option train_dataloader.dataset.split='sam10_VAE3_kmeans.txt' --work-dir work_dirs_det/yolov3-e20_samVAE10_uesat --amp
python DETtrain.py det_configs/yolo/yolov3_d53_8xb8-ms-608-273e_uesat.py --cfg-option train_dataloader.dataset.split='sam20_VAE3_kmeans.txt' --work-dir work_dirs_det/yolov3-e20_samVAE20_uesat --amp
python DETtrain.py det_configs/yolo/yolov3_d53_8xb8-ms-608-273e_uesat.py --work-dir work_dirs_det/yolov3-e20_full_uesat --amp
python DETtrain.py det_configs/detr/detr_r50_8xb2-150e_uesat.py --work-dir work_dirs_det/detr-e20_full_uesat --amp
