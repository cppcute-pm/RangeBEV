
dataset_cfgs=dict(
    data_type='oxford',
    dataset_path='Oxford_Robotcar',
    pc_dir='point_clouds_num_4096',
    rendered_dir='renderings_test',
    image_dir='real_rgb',
    mask_dir='image_mask',
    train_query_filename='class_split_train_queries_v2.pickle',
    val_query_filename='class_split_val_queries.pickle',
    eval_query_filename='class_split_test_queries.pickle',
    lidar2image_ndx_path='lidar2image_ndx.pickle',
    class_num_path='class_split_num_v2.pickle',
    image_size=[224, 224],
    render_size=[224, 224],
    mask_size=[224, 224],
    render_view_num=4,
    use_cloud=True,
    use_render=False,
    use_image=True,
    use_mask=False,
    num_points=None,
)

sampler_cfgs=dict(
    sampler_type='none',
)

batch_sampler_cfgs=dict(
    batch_sampler_type='ensure_pos',
    num_k=2,
    batch_size=16,
    batch_size_limit=16,
    batch_expansion_rate=1.5,
    eval_batch_size=32,
)

collect_fn_cfgs=dict(
    collect_fn_type='1',
)

dataloader_cfgs=dict(
    num_workers=8,
    dataset_cfgs=dataset_cfgs,
    sampler_cfgs=sampler_cfgs,
    batch_sampler_cfgs=batch_sampler_cfgs,
    collect_fn_cfgs=collect_fn_cfgs,
)