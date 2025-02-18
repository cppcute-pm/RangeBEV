_base_ = ['../base/models/cct.py',
            '../base/datasets/boreas.py',
            '../base/losses/v2_infonce.py',
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
    )
)

dataset_cfgs=dict(
    use_cloud=True,
    use_image=True,
    train_pc_aug_mode=0,
    num_points=4096,
    train_image_aug_mode=1,
    eval_pc_aug_mode=0,
    eval_image_aug_mode=3,
    pc_preprocess=dict(mode=0),
    use_overlap_ratio=True,
    ratio_strategy='mean',
    relative_strategy=True
)
sampler_cfgs=dict(sampler_type='none')
batch_sampler_cfgs=dict(
    batch_sampler_type='ensure_k_levels_pos',
    num_k=10,
    batch_size=16,
    safe_elem_region=50,
    eval_batch_size=16,
    iter_per_epoch=2000)
collect_fn_cfgs=dict(collect_fn_type='2')

dataloader_cfgs=dict(
    num_workers=16,
    sampler_cfgs=sampler_cfgs,
    batch_sampler_cfgs=batch_sampler_cfgs,
    collect_fn_cfgs=collect_fn_cfgs,
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
    lr=1e-5,
    weight_decay=1e-4,
)

scheduler_cfgs = dict(
    scheduler_type='CosineAnnealingWarmRestarts',
    warmup_epoch=5,
    min_lr=1e-7,
    T_mult=2)

loss_cfgs=dict(
    loss_type='v2_infonce',
    temperature=0.1, 
    reduction='mean',
    negative_mode=2, # TODO: 2、3
    positive_mode=3, # TODO: 2、3
    positive_overlap_margin=0.2,
    negative_overlap_margin=0.05,
    distance_mode=2,
)

freeze_cfgs=dict(
    freeze_imgnet=dict(
        backbone=0,
        )
)
use_mp=True
