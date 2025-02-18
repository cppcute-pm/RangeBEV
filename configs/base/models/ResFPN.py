
model_cfgs=dict(
    modal_type=1, # '1' '2' '3'
    out_dim=256,
    imgnet_cfgs=dict(
        backbone_type='ResFPN82',
        out_layer=1,
        backbone_config=dict(
            initial_dim=128,
            block_dims=[128, 196, 256],
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
        backbone=0,
        aggregate=0,
        )
)
pretrained_cfgs=dict(
    imgnet=dict(
        path='models/resnet50-19c8e357.pth',
        backbone_part=0,
        aggregate_part=None,
    )
)