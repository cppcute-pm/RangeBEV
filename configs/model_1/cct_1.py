
_base_ = ['../base/models/cct.py',
            '../base/datasets/oxford.py',
            '../base/losses/triplet.py',
            '../base/optimizers/adam.py',
            '../base/schedulers/CosineAnnealingWarmRestarts.py',
            '../base/runtime.py',]
model_cfgs=dict(
    imgnet_cfgs=dict(
        aggregate_type='GeM',
        out_layer=2,
        aggregate_config=dict(
            p=3,
            eps=1e-6,
        ),
        backbone_config=dict(
            f_layer=8,
            c_layer=14,
            img_size=224,
        ),
    )
)

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
    use_overlap_ratio=False,
    use_original_pc_correspondence=False,
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
    collect_fn_type='1',
)

dataloader_cfgs=dict(
    batch_sampler_cfgs=batch_sampler_cfgs,
    num_workers=8,
    dataset_cfgs=dataset_cfgs,
    sampler_cfgs=sampler_cfgs,
    collect_fn_cfgs=collect_fn_cfgs,
)

loss_cfgs=dict(
    loss_type='triplet',
    pair_dist_info=True,
    hard_mining=True,
    margin=0.3,
    normalize_embeddings=True,
)

cct_1_path="/home/pengjianyi/.cache/torch/hub/checkpoints/cct_14_7x2_224_imagenet.pth"
# cct_1_path=None
pretrained_cfgs=dict(
    imgnet=dict(
        path=cct_1_path,
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
    freeze_imgnet=dict(
        backbone=0,
        )
)

use_mp=True
