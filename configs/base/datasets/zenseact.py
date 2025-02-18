
dataset_cfgs=dict(
    data_type='zenseact',
    raw_name='zenseact',
    pc_name='zenseact_lidar_4096',
    num_points=4096,
    train_pc_aug_mode=4,
    eval_pc_aug_mode=0,
    use_render=False,
    use_cloud=True,
    use_image=True,
    train_image_aug_mode=1,
    eval_image_aug_mode=6,
    image_size=[224, 224],
    pc_preprocess=dict(
        mode=0,
    ),
)

sampler_cfgs=dict(
    sampler_type='distributed',
)

batch_sampler_cfgs=dict(
    batch_sampler_type='ordinary',
    batch_size=16,
    eval_batch_size=32,
    drop_last=True,
)

collect_fn_cfgs=dict(
    collect_fn_type='3',
)

dataloader_cfgs=dict(
    num_workers=8,
    dataset_cfgs=dataset_cfgs,
    sampler_cfgs=sampler_cfgs,
    batch_sampler_cfgs=batch_sampler_cfgs,
    collect_fn_cfgs=collect_fn_cfgs,
)