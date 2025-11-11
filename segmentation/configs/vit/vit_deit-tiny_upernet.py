_base_ = './vit_deit-tiny_mln_upernet.py'

model = dict(
    pretrained='',
    backbone=dict(drop_path_rate=0.1),
    neck=None)
