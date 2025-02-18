_base_ = ['../base/datasets/oxford.py',
            '../base/losses/triplet.py',
            '../base/optimizers/adam.py',
            '../base/schedulers/CosineAnnealingWarmRestarts.py',
            '../base/runtime.py',]
model_cfgs=dict(
    modal_type=3,
    out_dim=256,
    cmvpr_cfgs=dict(
        image_encoder_type='CCT',
        image_encoder_out_layer=3,
        image_encoder_cfgs=dict(
            f_layer=8,
            c_layer=8,
            img_size=224,
            ),
        pc_encoder_type='DGCNN',
        pc_encoder_out_layer=3,
        pc_encoder_cfgs=dict(
            out_k=8,
            k=8,
            emb_dims=256,
            dropout=0.5,
            f_layer=2,
            c_layer=5,
            reducer=[2, 2, 2, 2, 2]
            ),
        image_aggregator_type='GeM',
        image_aggregator_cfgs=dict(
            p=3,
            eps=1e-6,
            ),
        pc_aggregator_type='GeM',
        pc_aggregator_cfgs=dict(
            p=3,
            eps=1e-6,
            ),
        phase=1,
    ),
)

dataset_cfgs=dict(
    num_points=4096,
    use_overlap_ratio=False,
)

batch_sampler_cfgs=dict(
    batch_sampler_type='ensure_k_pos',
    num_k=2,
    batch_size=16,
    batch_size_limit=16,
    batch_expansion_rate=1.5,
    eval_batch_size=16,
)

dataloader_cfgs=dict(
    dataset_cfgs=dataset_cfgs,
    batch_sampler_cfgs=batch_sampler_cfgs,
    num_workers=8,
)

cct_path="9245wsl0/epoch_3.pth"
# cct_path=None
# dgcnn_path=None
dgcnn_path="bdwkjsik/epoch_59.pth"
loss_cfgs=dict(
    loss_type='triplet_cm',
    pair_dist_info=True,
    hard_mining=True,
    margin=0.2,
    normalize_embeddings=True,
)

optimizer_cfgs=dict(
    optimizer_type='adam',
    lr=1e-5,
    # image_encoder_lr=1e-5,
    # image_aggregator_lr=1e-4,
    # pc_encoder_lr=1e-4,
    # pc_aggregator_lr=1e-4,
    weight_decay=1e-4,
)

scheduler_cfgs=dict(
    scheduler_type='CosineAnnealingWarmRestarts',
    warmup_epoch=5,
    min_lr=1e-9,
    T_mult=2,
)

pretrained_cfgs=dict(
    cmvprnet=dict(
        image_path=cct_path,
        image_cfgs=dict(
            backbone_part=0
        ),
        image_aggregator_cfgs=dict(
            aggregate_part=0
        ),
        pc_path=dgcnn_path,
        pc_cfgs=dict(
            backbone_part=0
        ),
        pc_aggregator_cfgs=dict(
            aggregate_part=0
        ),
    )
)

freeze_cfgs=dict(
    freeze_cmvpr=dict(
        freeze_img_encoder=dict(
            backbone=0,
        ),
        freeze_img_aggregator=dict(
            aggregate=0,
        ),
        freeze_pc_encoder=dict(
            backbone=0,
        ),
        freeze_pc_aggregator=dict(
            aggregate=0,
        ),
    ),
)

evaluator_type=3
use_mp=True
find_unused_parameters=False
all_gather_cfgs=dict(
    all_gather_flag=False,
    types=['embeddings1', 'embeddings2', 'positives_mask', 'negatives_mask']
)
accumulate_iter=1
