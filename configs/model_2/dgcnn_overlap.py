_base_ = ['../base/models/dgcnn.py',
            '../base/datasets/boreas.py',
            '../base/losses/triplet.py',
            '../base/optimizers/adam.py',
            '../base/schedulers/CosineAnnealingWarmRestarts.py',
            '../base/runtime.py',]

model_cfgs=dict(
    pcnet_cfgs=dict(
        backbone_config=dict(
            out_k=20,
            k=20,
            emb_dims=256,
        ),
    ),
)

dataloader_cfgs = dict(
    num_workers=0,
    dataset_cfgs=dict(
        train_pc_aug_mode=5,
        num_points=4096,
        train_image_aug_mode=9,
        eval_pc_aug_mode=0,
        eval_image_aug_mode=3,
        data_type='boreas',
        raw_dir_name='Boreas_minuse',
        pc_dir_name='Boreas_lidar_4096',
        rendered_dir_name=None,
        image_dir_name=None,
        mask_dir_name=None,
        tool_name='my_tool',
        train_query_filename='class_split_train_queries.pickle',
        val_query_filename='class_split_test_queries.pickle',
        eval_query_filename='class_split_test_queries.pickle',
        lidar2image_filename='lidar2image.pickle',
        image_size=[224, 224],
        mask_size=[224, 224],
        render_size=[224, 224],
        pc_preprocess=dict(mode=0),
        render_view_num=None,
        use_cloud=True,
        use_render=False,
        use_image=True,
        use_mask=False,
        ratio_strategy='mean',
        relative_strategy=True,
        use_overlap_ratio=False,),
    sampler_cfgs=dict(sampler_type='none'),
    batch_sampler_cfgs=dict(
        batch_sampler_type='ensure_k_levels_pos',
        num_k=10,
        batch_size=16,
        eval_batch_size=32,
        iter_per_epoch=2000,),
    # batch_sampler_cfgs=dict(
    #     batch_sampler_type='ensure_k_levels_pos',
    #     num_k=10,
    #     batch_size=16,
    #     safe_elem_region=50,
    #     iter_per_epoch=4000,
    #     eval_batch_size=32,
    # ),
    collect_fn_cfgs=dict(collect_fn_type='2'))
# loss_cfgs = dict(
#     loss_type='v2_triplet',
#     base_margin=0.2,
#     normalize_embeddings=False,
#     positive_overlap_ratio=0.2,
#     negative_overlap_ratio=0.05,
#     delta_overlap_ratio_margin=0.01,
#     tuple_formtype='relative_delta')
# dgcnn_1_path="/home/pengjianyi/.cache/torch/hub/checkpoints/dgcnn_model_1024_t7.pth"
dgcnn_1_path=None
pretrained_cfgs=dict(
    pcnet=dict(
        path=dgcnn_1_path,
        backbone_part=0,
    )
)
freeze_cfgs=dict(
    freeze_pcnet=dict(
        backbone=0, # TODO: backbone=2
        )
)