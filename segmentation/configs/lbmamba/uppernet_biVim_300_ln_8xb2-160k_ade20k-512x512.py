_base_ = './uppernet_biVim_tiny_ln_8xb2-160k_ade20k-512x512.py'


model = dict(
    pretrained=None,
    backbone=dict(
        pretrained_ckpt='',
        embed_dim=300,
        out_dim=192 # adjust dim to 192, consistent with tiny version
    ),
    decode_head=dict(
        in_channels=[192, 192, 192, 192]
    ),
    auxiliary_head=dict(
        in_channels=192
    ),
)