_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8-n11-ft.py',
    '../_base_/datasets/uesatRGB10.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_uesat.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
# model = dict(data_preprocessor=data_preprocessor)
model = dict(data_preprocessor=data_preprocessor,
             pretrained='work_dirs/deeplab_e5-b8_uesat11_VAEn_5/epoch_5.pth')
# train_cfg = dict(type='EpochBasedTrainLoop',  max_epochs=5, val_interval=1)
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        split="HIL5.txt"))
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    drop_last=True,
    dataset=dict(
        data_root='data/HIL_resized/',
        data_prefix=dict(
            img_path='images/validation', 
            seg_map_path='annotations/validation'),
        split="val.txt",
))
test_dataloader = val_dataloader
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)