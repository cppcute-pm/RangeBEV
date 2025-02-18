dataset_cfgs=dict(
    data_type='boreas_v2',
    train_pc_aug_mode=5,
    eval_pc_aug_mode=0,
    train_image_aug_mode=1,
    eval_image_aug_mode=3,
    raw_dir_name='Boreas_minuse',
    pc_dir_name='Boreas_lidar_4096',
    rendered_dir_name=None,
    image_dir_name=None,
    mask_dir_name=None,
    tool_name='my_tool',
    train_coords_filename='train_UTM_coords.npy',
    val_query_filename='class_split_val_queries.pickle',
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
)

sampler_cfgs=dict(
    sampler_type='none',
)

batch_sampler_cfgs=dict(
    batch_sampler_type='ensure_k_levels_pos_v2',
    num_k=10,
    batch_size=16,
    safe_elem_region=50,
    iter_per_epoch=2000,
    eval_batch_size=32,
    level1_distance=10.0,
)

collect_fn_cfgs=dict(
    collect_fn_type='2',
)

dataloader_cfgs=dict(
    num_workers=8,
    dataset_cfgs=dataset_cfgs,
    sampler_cfgs=sampler_cfgs,
    batch_sampler_cfgs=batch_sampler_cfgs,
    collect_fn_cfgs=collect_fn_cfgs,
)