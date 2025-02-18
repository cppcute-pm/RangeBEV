
_base_ = ['../base/models/cct.py',
            '../base/datasets/boreas.py',
            '../base/losses/triplet.py',
            '../base/optimizers/adam.py',
            '../base/schedulers/CosineAnnealingWarmRestarts.py',
            '../base/runtime.py',]
model_cfgs=dict(
    imgnet_cfgs=dict(
        aggregate_type='GeM',
        out_layer=2,
        aggregate_config=dict(
            p=3,
            eps=1e-6,
        ),
        backbone_config=dict(
            f_layer=8,
            c_layer=14,
            img_size=224,
        ),
    )
)

dataset_cfgs=dict(
    use_cloud=False,
    use_image=True,
    use_overlap_ratio=False,
)

dataloader_cfgs=dict(
    num_workers=16,
    batch_sampler_cfgs=dict(
        batch_sampler_type='ensure_k_pos',
        batch_size=16,
        batch_size_limit=16,
        batch_expansion_rate=1.5,
        num_k=2,

        eval_batch_size=16,
    ),
    collect_fn_cfgs=dict(
        collect_fn_type='1',
    ),
    dataset_cfgs=dataset_cfgs,
)
cct_1_path="/home/test5/.cache/torch/hub/checkpoints/cct_14_7x2_224_imagenet.pth"
# cct_1_path=None
pretrained_cfgs=dict(
    imgnet=dict(
        path=cct_1_path,
    )
)

optimizer_cfgs=dict(
    optimizer_type='adam',
    lr=1e-7,
    weight_decay=1e-4,
)

scheduler_cfgs=dict(
    scheduler_type='CosineAnnealingWarmRestarts',
    warmup_epoch=5,
    min_lr=1e-9,
    T_mult=2,
)

freeze_cfgs=dict(
    freeze_imgnet=dict(
        backbone=2,
        )
)
