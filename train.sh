#!/bin/bash
pip install -e /root/code/ALSAM/
python /root/code/ALSAM/MMtrain.py /root/code/ALSAM/configs/mask2former/mask2former_swin-b-in1k-384x384-pre_coco.py   --work-dir /mnt/work_dirs/mask2former_full_coco --amp