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


dataloader_cfgs = dict(
    num_workers=8,
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
        use_image=False,
        use_mask=False,
        num_points=None,
        use_overlap_ratio=False,),
    sampler_cfgs=dict(sampler_type='none'),
    batch_sampler_cfgs=dict(
        batch_sampler_type='ensure_pos',
        num_k=2,
        batch_size=16,
        batch_size_limit=16,
        batch_expansion_rate=1.5,
        eval_batch_size=16),
    collect_fn_cfgs=dict(collect_fn_type='1'),
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