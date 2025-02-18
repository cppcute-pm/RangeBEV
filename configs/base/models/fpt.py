

model_cfgs=dict(
    modal_type=2, # '1' '2' '3'
    out_dim=256,
    pcnet_cfgs=dict(
        backbone_type='FPT',
        out_layer=1,
        backbone_config=dict(
            in_channels=3,
            out_channels=256,
            f_layer_num=4,
            radius_max_raw=60,
            voxel_size=0.3,
            ),
        aggregate_type='GeM',
        aggregate_config=dict(
            p=3,
            eps=1e-6,
        ),
    ),
)

freeze_cfgs=dict(
    freeze_pcnet=dict(
        backbone=0,
        aggregate=0,
        )
)
pretrained_cfgs=dict(
    pcnet=dict(
        path=None,
        backbone_part=None,
        aggregate_part=None,
    )
)