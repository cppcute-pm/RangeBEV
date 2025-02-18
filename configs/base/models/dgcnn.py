
model_cfgs=dict(
    modal_type=2, # '1' '2' '3'
    out_dim=256,
    pcnet_cfgs=dict(
        backbone_type='DGCNN',
        out_layer=1,
        backbone_config=dict(
            out_k=20,
            k=20,
            emb_dims=256,
            dropout=0.5,
            f_layer=2,
            c_layer=5,
            reducer=[2, 2, 2, 2, 2]
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