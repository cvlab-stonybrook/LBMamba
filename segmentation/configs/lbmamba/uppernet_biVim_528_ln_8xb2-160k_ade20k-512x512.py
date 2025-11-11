_base_ = './uppernet_biVim_tiny_ln_8xb2-160k_ade20k-512x512.py'


model = dict(
    pretrained=None,
    backbone=dict(
        pretrained_ckpt=None,
        embed_dim=528,
        out_dim=384 # adjust dim to 192, consistent with tiny version
    ),
    decode_head=dict(
        in_channels=[384, 384, 384, 384]
    ),
    auxiliary_head=dict(
        in_channels=384
    ),
)