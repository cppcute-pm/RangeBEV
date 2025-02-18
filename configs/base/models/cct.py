

model_cfgs=dict(
    modal_type=1, # '1' '2' '3'
    out_dim=256,
    imgnet_cfgs=dict(
        backbone_type='CCT',
        out_layer=1,
        backbone_config=dict(
            f_layer=8,
            c_layer=14,
            img_size=224,
        ),
        aggregate_type='GeM',
        aggregate_config=dict(
            p=3,
            eps=1e-6,
        ),
    ),
    )

freeze_cfgs=dict(
    freeze_imgnet=dict(
        backbone=5,
        aggregate=0,
        )
)
pretrained_cfgs=dict(
    imgnet=dict(
        path="~/.cache/torch/hub/checkpoints/cct_14_7x2_224_imagenet.pth",
        backbone_part=0,
        aggregate_part=None,
    )
)