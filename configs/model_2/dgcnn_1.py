_base_ = ['../base/models/dgcnn.py',
            '../base/datasets/oxford.py',
            '../base/losses/triplet.py',
            '../base/optimizers/adam.py',
            '../base/schedulers/CosineAnnealingWarmRestarts.py',
            '../base/runtime.py',]

model_cfgs=dict(
    pcnet_cfgs=dict(
        backbone_config=dict(
            out_k=8,
            k=8,
            emb_dims=256,
        ),
    ),
)


dataset_cfgs=dict(
    use_cloud=True,
    use_image=False,
)
dataloader_cfgs=dict(
    batch_sampler_cfgs=dict(
        batch_sampler_type='ensure_k_pos',
        batch_size=16,
        batch_size_limit=16,
        eval_batch_size=16,
        num_k=2,
    ),
    dataset_cfgs=dataset_cfgs,
)

# dgcnn_1_path="/home/pengjianyi/.cache/torch/hub/checkpoints/dgcnn_model_1024_t7.pth"
dgcnn_1_path=None
pretrained_cfgs=dict(
    pcnet=dict(
        path=dgcnn_1_path,
        backbone_part=0,
    )
)

optimizer_cfgs=dict(
    optimizer_type='adam',
    lr=1e-5,
    pc_backbone_lr=1e-5,
    pc_aggregator_lr=1e-4,
    weight_decay=1e-4,
)

scheduler_cfgs=dict(
    scheduler_type='CosineAnnealingWarmRestarts',
    warmup_epoch=5,
    min_lr=1e-7,
    T_mult = 2,
)

freeze_cfgs=dict(
    freeze_pcnet=dict(
        backbone=0, # TODO: backbone=2
        )
)

use_mp=True
accumulate_iter=1