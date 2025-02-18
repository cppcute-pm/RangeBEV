_base_ = ['../base/datasets/boreas_v2.py',
            '../base/losses/v2_triplet.py',
            '../base/optimizers/adam.py',
            '../base/schedulers/CosineAnnealingWarmRestarts.py',
            '../base/runtime.py',]
model_cfgs=dict(
    modal_type=5,
    out_dim=256,
    cmvpr2_cfgs=dict(
        image_encoder_type='CCT',
        image_encoder_out_layer=3,
        image_out_layer=2,
        render_out_layer=2,
        image_encoder_cfgs=dict(
            f_layer=8,
            c_layer=14,
            img_size=224,
            ),
        render_encoder_type='CCT',
        render_encoder_out_layer=3,
        render_encoder_cfgs=dict(
            f_layer=8,
            c_layer=14,
            img_size=224,
            ),
        image_aggregator_type='GeM',
        image_aggregator_cfgs=dict(
            p=3,
            eps=1e-6,
            ),
        render_aggregator_type='GeM',
        render_aggregator_cfgs=dict(
            p=3,
            eps=1e-6,
            ),
        phase=1,
    ),
)

dataset_cfgs=dict(
    train_render_aug_mode=14,
    eval_render_aug_mode=14,
    train_image_aug_mode=11,
    eval_image_aug_mode=14,
    train_pc_aug_mode=0,
    eval_pc_aug_mode=0,
    raw_dir_name='Boreas_minuse',
    pc_dir_name="Boreas_minuse_40960_to_4096_cliped_fov",
    # pc_dir_name=None,
    rendered_dir_name='Boreas_minuse_lidar_depth',
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
    use_render=True,
    use_image=True,
    use_mask=False,
    num_points=4096,
    ratio_strategy='min',
    relative_strategy=True,
    use_overlap_ratio=True,
    use_original_pc_correspondence=False,
    point_project_threshold=150.0,
    crop_location=dict(
        x_min=0,
        x_max=2448,
        y_min=683,
        y_max=1366,
    ),
)

batch_sampler_cfgs=dict(
    batch_sampler_type='ensure_k_pos',
    num_k=2,
    batch_size=16,
    batch_size_limit=16,
    batch_expansion_rate=1.5,
    eval_batch_size=16,
)

sampler_cfgs=dict(
    sampler_type='none',
)

collect_fn_cfgs=dict(
    collect_fn_type='4',
)

dataloader_cfgs=dict(
    batch_sampler_cfgs=batch_sampler_cfgs,
    num_workers=8,
    dataset_cfgs=dataset_cfgs,
    sampler_cfgs=sampler_cfgs,
    collect_fn_cfgs=collect_fn_cfgs,
)

cct1_path="hsp6sjly/epoch_79.pth"
cct2_path="n8reg0zb/epoch_55.pth"

loss_cfgs = dict(
    loss_type='v2_triplet_cm',
    base_margin=0.6,
    normalize_embeddings=False,
    positive_overlap_ratio=0.2,
    negative_overlap_ratio=0.05,
    delta_overlap_ratio=0.01,
    tuple_formtype='relative_delta',
    choose_nonzero=True)

pretrained_cfgs=dict(
    cmvpr2net=dict(
        image_path=cct1_path,
        image_cfgs=dict(
            backbone_part=0
        ),
        image_aggregator_cfgs=dict(
            aggregate_part=0
        ),
        render_path=cct2_path,
        render_cfgs=dict(
            backbone_part=0
        ),
        render_aggregator_cfgs=dict(
            aggregate_part=0
        ),
    )
)

freeze_cfgs=dict(
    freeze_cmvpr2=dict(
        freeze_img_encoder=dict(
            backbone=13,
        ),
        freeze_img_aggregator=dict(
            aggregate=0,
        ),
        freeze_render_encoder=dict(
            backbone=13,
        ),
        freeze_render_aggregator=dict(
            aggregate=0,
        ),
    )
)

use_mp=True
# find_unused_parameters=False
# all_gather_cfgs=dict(
#     all_gather_flag=True,
#     types=['embeddings1', 'embeddings2', 'positives_mask', 'negatives_mask']
# )
# accumulate_iter=8 
