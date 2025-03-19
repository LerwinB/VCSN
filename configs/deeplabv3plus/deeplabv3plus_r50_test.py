_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/uesatRGB.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_uesat.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
val_dataloader = dict(
    dataset=dict(
        data_prefix=dict(
            img_path='images/test1',
            seg_map_path='annotations/test1'),
))
test_dataloader = val_dataloader