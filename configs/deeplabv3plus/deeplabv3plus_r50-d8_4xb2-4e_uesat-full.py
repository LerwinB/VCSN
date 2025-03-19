_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/uesat1000x1000MM.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_uesat.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
