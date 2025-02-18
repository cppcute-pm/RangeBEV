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
            f_layer=7,
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
    use_overlap_ratio=False,
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

optimizer_cfgs=dict(
    optimizer_type='adam',
    lr=1e-5,
    image_backbone_lr=1e-5,
    image_aggregator_lr=1e-5,
    weight_decay=1e-4,
)

freeze_cfgs=dict(
    freeze_imgnet=dict(
        backbone=8,
        )
)
use_mp=True