_base_ = ['../base/models/cct.py',
            '../base/datasets/oxford.py',
            '../base/losses/triplet.py',
            '../base/optimizers/adam.py',
            '../base/schedulers/CosineAnnealingWarmRestarts.py',
            '../base/runtime.py',]
model_cfgs=dict(
    imgnet_cfgs=dict(
        out_layer=2,
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
    )
)

dataset_cfgs=dict(
    use_cloud=False,
    use_image=True,
)

dataloader_cfgs=dict(
    batch_sampler_cfgs=dict(
        batch_sampler_type='ensure_k_pos',
        batch_size=16,
        batch_size_limit=16,
        num_k=2,
    ),
    dataset_cfgs=dataset_cfgs,
)
cct_1_path="/home/pengjianyi/.cache/torch/hub/checkpoints/cct_14_7x2_224_imagenet.pth"
# cct_1_path=None
pretrained_cfgs=dict(
    imgnet=dict(
        path=cct_1_path,
    )
)

freeze_cfgs=dict(
    freeze_imgnet=dict(
        backbone=0,
        )
)

loss_cfgs=dict(
    loss_type='triplet',
    pair_dist_info=True,
    hard_mining=True,
    margin=0.2,
    normalize_embeddings=True,
)

optimizer_cfgs=dict(
    optimizer_type='adam',
    lr=1e-5,
    # image_backbone_lr=1e-7,
    # image_aggregator_lr=1e-5,
    weight_decay=1e-4,
)

scheduler_cfgs=dict(
    scheduler_type='CosineAnnealingWarmRestarts',
    warmup_epoch=5,
    min_lr=1e-7,
    T_mult=2,
)

use_mp=True
evaluator_type=3
