#!/bin/bash
source ~/anaconda3/bin/activate openmmlab;
export CUDA_VISIBLE_DEVICES=0;
echo "wait 4000s";
sleep 4000;
# 要监测的进程名称或PID
pid="1999161"  # 可以替换为进程名或PID

# 定义要执行的下一个任务
next_task() {
    echo "开始执行下一个任务..."
    python pt_active_select.py --dataset_name uesat --task_name Coreset --data_pre False;
    # 在这里添加你的任务，例如启动另一个进程或执行某个命令
    # 示例：/path/to/your/command
}

# 监测进程状态
while true; do
    if ps -p $pid > /dev/null 2>&1; then
        echo "进程 $pid 仍在运行"
        sleep 300  # 每隔5秒检查一次，可以根据需要调整
    else
        echo "进程 $pid 已结束，开始执行下一个任务..."
        next_task
        break  # 结束循环
    fi
done

echo "pt_active_select.py completed";
python MMtrain.py configs/deeplabv3plus/deeplabv3plus_r50_SAM5_uesatrgb.py --cfg-options train_dataloader.dataset.split="Coreset_5_kmeans.txt" --work-dir "work_dirs/deeplabv3plus_r50_Coreset5" --amp;
# 1025
# python CLStrain.py --modelname 'Resnet' --data_list 'Entropy5.txt' --work_dir './work_dirs/Resnet5_Entropy_uesat';
# python CLStrain.py --modelname 'Resnet' --data_list 'resnet5_kmeans.txt' --work_dir './work_dirs/Resnet5_coreset_uesat';
# python CLStrain.py --modelname 'Resnet' --data_list 'ULAL5.txt' --work_dir './work_dirs/Resnet5_ULAL_uesat';
# python DETtrain.py det_configs/yolo/yolov3_d53_8xb8-ms-608-273e_uesat.py --cfg-option train_dataloader.dataset.split='Entropy5.txt' --work-dir work_dirs_det/yolov3-e20_Entropy5_uesat --amp;
# python DETtrain.py det_configs/yolo/yolov3_d53_8xb8-ms-608-273e_uesat.py --cfg-option train_dataloader.dataset.split='resnet5_kmeans.txt' --work-dir work_dirs_det/yolov3-e20_coreset5_uesat --amp;
# python DETtrain.py det_configs/yolo/yolov3_d53_8xb8-ms-608-273e_uesat.py --cfg-option train_dataloader.dataset.split='ULAL5.txt' --work-dir work_dirs_det/yolov3-e20_ULAL5_uesat --amp;