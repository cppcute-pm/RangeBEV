_base_ = [
            '../base/datasets/boreas_v2.py',
            '../base/losses/triplet.py',
            '../base/optimizers/adam.py',
            '../base/schedulers/CosineAnnealingWarmRestarts.py',
            '../base/runtime.py',]

model_cfgs=dict(
    modal_type=2,
    out_dim=256,
    pcnet_cfgs=dict(
        backbone_type='PointNextv2',
        out_layer=4,
        backbone_config=dict(
            in_channels=4,
            width=32,
            radius=3.5,
            nsample=32,
            dropout=0.5,
            pointnextv3_op=1,
        ),
        aggregate_type='PoS_GeM',
        aggregate_config=dict(
            p=[3, 3, 3, 3, 3],
            eps=[1e-6, 1e-6, 1e-6, 1e-6, 1e-6],
            type='type_4',
        ),
        coords_type='backbone',
    ),
)

dataset_cfgs=dict(
    train_pc_aug_mode=8,
    eval_pc_aug_mode=3,
    train_image_aug_mode=16,
    eval_image_aug_mode=17,
    raw_dir_name='Boreas_minuse',
    pc_dir_name="Boreas_minuse_40960_to_4096_cliped_fov",
    # pc_dir_name=None,
    rendered_dir_name=None,
    image_dir_name='Boreas_224x224_image',
    mask_dir_name=None,
    tool_name='my_tool',
    train_coords_filename='train_UTM_coords_v3_50m.npy',
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
    use_image=False,
    use_mask=False,
    num_points=4096,
    crop_location=dict(
        x_min=0,
        x_max=2448,
        y_min=683,
        y_max=1366,
    ),
    img_neighbor_num=1,
    dist_caculation_type='all_coords_L2_mean',
    use_overlap_ratio=True,
    overlap_ratio_type='pose_dist_sim',
    pose_dist_threshold=37.5,
    use_original_pc_correspondence=False,
    correspondence_point_project_threshold=110.0,
)


# batch_sampler_cfgs=dict(
#     batch_sampler_type='ensure_k_pos',
#     num_k=2,
#     batch_size=16,
#     batch_size_limit=16,
#     batch_expansion_rate=1.5,
#     eval_batch_size=16,
# )

batch_sampler_cfgs=dict(
    batch_sampler_type='ensure_k_levels_pos_uniform_multi_v8',
    num_k=3,
    batch_size=16,
    eval_batch_size=16,
    interval_num=50,
    useout_times=1,
)

sampler_cfgs=dict(
    sampler_type='none',
)

collect_fn_cfgs=dict(
    collect_fn_type='4',
)

dataloader_cfgs=dict(
    batch_sampler_cfgs=batch_sampler_cfgs,
    num_workers=4,
    dataset_cfgs=dataset_cfgs,
    sampler_cfgs=sampler_cfgs,
    collect_fn_cfgs=collect_fn_cfgs,
)

# loss_cfgs=dict(
#     loss_type='triplet',
#     pair_dist_info=True,
#     hard_mining=True,
#     margin=0.2,
#     normalize_embeddings=False,
# )

loss_cfgs = dict(
    loss_type='v2_triplet',
    base_margin=0.6,
    normalize_embeddings=False,
    positive_overlap_ratio=0.2,
    negative_overlap_ratio=0.05,
    delta_overlap_ratio=0.01,
    tuple_formtype='relative_delta',
    choose_nonzero=True)


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
# pointnext_path="/home/pengjianyi/.cache/torch/hub/checkpoints/s3dis-train-pointnext-b_v2.pth"
pointnext_path=None
pretrained_cfgs=dict(
    pcnet=dict(
        path=pointnext_path,
        backbone_part=0,
        aggregate_part=None,
    )
)
use_mp=True
need_eval=True
freeze_cfgs=dict(
    freeze_pcnet=dict(
        backbone=2,
        aggregate=0,
        )
)