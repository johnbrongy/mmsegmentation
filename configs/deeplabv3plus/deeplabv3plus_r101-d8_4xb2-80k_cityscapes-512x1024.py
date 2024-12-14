_base_ = './deeplabv3plus_r50-d8_4xb2-80k_cityscapes-512x1024.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101, cbam_ratio = 8, cbam_kernel_size = 7))

checkpoint_config = dict(interval=4000, max_keep_ckpts=10)
work_dir = '/content/drive/MyDrive/deeplabv3+_cbam_natocc'
