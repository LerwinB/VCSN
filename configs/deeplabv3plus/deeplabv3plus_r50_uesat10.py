_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8-n11.py',
    '../_base_/datasets/uesatRGB10.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_uesat.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
# train_cfg = dict(type='EpochBasedTrainLoop',  max_epochs=5, val_interval=1)
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        split="VAE3n+HIL5.txt"))
