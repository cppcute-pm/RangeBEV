_base_ = ['../base/datasets/boreas_v2.py',
            '../base/losses/v2_triplet.py',
            '../base/optimizers/adam.py',
            '../base/schedulers/CosineAnnealingWarmRestarts.py',
            '../base/runtime.py',]
model_cfgs=dict(
    modal_type=3,
    out_dim=256,
    cmvpr_cfgs=dict(
        image_encoder_type='CCT',
        image_out_layer=2,
        pc_out_layer=1,
        image_encoder_out_layer=3,
        image_encoder_cfgs=dict(
            f_layer=8,
            c_layer=14,
            img_size=224,
            ),
        pc_encoder_type='DGCNN',
        pc_encoder_out_layer=3,
        pc_encoder_cfgs=dict(
            k=20,
            out_k=20,
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
        phase=4,
    ),
)

dataset_cfgs=dict(
    train_pc_aug_mode=5,
    eval_pc_aug_mode=0,
    train_image_aug_mode=1,
    eval_image_aug_mode=3,
    raw_dir_name='Boreas_minuse',
    pc_dir_name="Boreas_minuse_163840_to_4096",
    rendered_dir_name=None,
    image_dir_name=None,
    mask_dir_name=None,
    tool_name='my_tool',
    train_coords_filename='train_UTM_coords.npy',
    val_query_filename='class_split_test_queries.pickle',
    eval_query_filename='class_split_test_queries.pickle',
    minuse_lidar_filename='train_minuse_lidar_idx.pickle',
    positive_distance=10.0,
    non_negative_distance=50.0,
    positive_distance_list=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
    lidar2image_filename='lidar2image.pickle',
    class_num_path='class_split_num.pickle',
    image_size=[224, 224],
    render_size=[224, 224],
    mask_size=[224, 224],
    pc_preprocess=dict(
        mode=0,
    ),
    use_cloud=True,
    use_render=False,
    use_image=True,
    use_mask=False,
    num_points=4096,
    ratio_strategy='mean',
    relative_strategy=True,
    use_overlap_ratio=True,
    use_original_pc_correspondence=False,
    point_project_threshold=200.0,
)

sampler_cfgs=dict(
    sampler_type='none',
)

batch_sampler_cfgs=dict(
    batch_sampler_type='ensure_k_levels_pos_uniform_multi_v2',
    num_k=10,
    batch_size=22,
    iter_per_epoch=2000,
    eval_batch_size=22,
)

collect_fn_cfgs=dict(
    collect_fn_type='4',
)

dataloader_cfgs=dict(
    num_workers=11,
    dataset_cfgs=dataset_cfgs,
    sampler_cfgs=sampler_cfgs,
    batch_sampler_cfgs=batch_sampler_cfgs,
    collect_fn_cfgs=collect_fn_cfgs,
)


optimizer_cfgs=dict(
    optimizer_type='adam',
    lr=1e-5,
    weight_decay=1e-4,
)

scheduler_cfgs=dict(
    scheduler_type='CosineAnnealingWarmRestarts',
    warmup_epoch=5,
    min_lr=1e-7,
    T_mult=2,
)

cct_path="rd3q6zdk/epoch_73.pth"
dgcnn_path="4jhsl2hh/epoch_69.pth"
# cct_path=None
# dgcnn_path=None
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
        cmvpr_path=None,
    )
)

freeze_cfgs=dict(
    freeze_cmvpr=dict(
        freeze_img_encoder=dict(
            backbone=13,
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
    )
)

loss_cfgs=dict(
    loss_type='GL_v2_triplet_circle',
    global_loss_cfgs=dict(
        base_margin=0.6,
        normalize_embeddings=False,
        positive_overlap_ratio=0.2,
        negative_overlap_ratio=0.05,
        delta_overlap_ratio=0.01,
        tuple_formtype='relative_delta',
        choose_nonzero=True,
    ),
    local_loss_cfgs=dict(
        m=0.4,
        gamma=80,
    ),
    global_loss_weight=0.5,
    local_loss_weight=0.5,
    local_positive_mask_margin=0.2,
    local_negative_mask_margin=0.01,
)

use_mp = True