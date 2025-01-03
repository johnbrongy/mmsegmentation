_base_ = './deeplabv3plus_r50-d8_4xb2-80k_cityscapes-512x1024.py'
model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        depth=101,
        dilations=(1, 1, 1, 2),
        strides=(1, 2, 2, 1),
        multi_grid=(1, 2, 4)),
        cbam_ratio=8,
        cbam_kernel_size=7)
    decode_head=dict(
        dilations=(1, 6, 12, 18),
        sampler=dict(type='OHEMPixelSampler', min_kept=100000)))
