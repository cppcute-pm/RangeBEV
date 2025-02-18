_base_ = ['../base/datasets/zenseact.py',
            '../base/losses/infonce.py',
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
            c_layer=14,
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
    pc_name="zenseact_lidar4096",
    use_cloud=True,
    use_image=True,
    # train_pc_aug_mode=5,
    # train_image_aug_mode=1,
    train_pc_aug_mode=1,
    train_image_aug_mode=1,
    eval_pc_aug_mode=0,
    eval_image_aug_mode=3,
    pc_preprocess=dict(
        mode=0,
        # voxel_size=0.5,
        # num_points=4096,
    ),
)

sampler_cfgs=dict(
    sampler_type='random',
)

batch_sampler_cfgs=dict(
    batch_sampler_type='ordinary',
    batch_size=32,
    eval_batch_size=32,
    drop_last=True,
)

dataloader_cfgs=dict(
    batch_sampler_cfgs=batch_sampler_cfgs,
    sampler_cfgs=sampler_cfgs,
    num_workers=16,
    dataset_cfgs=dataset_cfgs,
)

cct_path="/home/pengjianyi/.cache/torch/hub/checkpoints/cct_14_7x2_224_imagenet.pth"
# cct_path=None
dgcnn_path=None
loss_cfgs=dict(
    loss_type='infonce_cm',
    temperature=0.001, 
    reduction='mean',
    negative_mode=1, 
    positive_mode=1, 
    distance_mode=2,
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
            backbone=5,
        ),
        freeze_img_aggregator=dict(
            aggregate=0,
        ),
        freeze_pc_encoder=dict(
            backbone=3,
        ),
        freeze_pc_aggregator=dict(
            aggregate=0,
        ),
    )
)

use_mp=True
find_unused_parameters=False
all_gather_cfgs=dict(
    all_gather_flag=False,
    types=['embeddings1', 'embeddings2', 'positives_mask', 'negatives_mask']
)
accumulate_iter=8
