_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8-n11.py',
    '../_base_/datasets/HIL.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_uesat.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(data_preprocessor=data_preprocessor)
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        split="HIL_SAMVAE_5_kmeans.txt"))