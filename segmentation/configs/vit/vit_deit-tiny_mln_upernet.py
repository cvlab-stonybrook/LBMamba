_base_ = './vit_vit-b16_mln_upernet_8xb2-160k_ade20k-512x512.py'

model = dict(
    pretrained='',
    backbone=dict(num_heads=3, embed_dims=192, drop_path_rate=0.1),
    decode_head=dict(num_classes=150, in_channels=[192, 192, 192, 192]),
    neck=dict(in_channels=[192, 192, 192, 192], out_channels=384),
    auxiliary_head=dict(num_classes=150, in_channels=192))
