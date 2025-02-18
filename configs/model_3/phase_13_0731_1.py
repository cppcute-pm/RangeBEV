_base_ = ['../base/datasets/boreas_v2.py',
            '../base/optimizers/adam.py',
            '../base/schedulers/CosineAnnealingWarmRestarts.py',
            '../base/runtime.py',]
model_cfgs=dict(
    modal_type=3,
    out_dim=256,
    cmvpr_cfgs=dict(
        image_encoder_type='ResUNetmmsegv3',
        image_out_layer=None,
        pc_out_layer=None,
        image_encoder_out_layer=9,
        image_encoder_cfgs=dict(),
        pc_encoder_type='PointNextv4',
        pc_encoder_out_layer=9,
        pc_encoder_cfgs=dict(),
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
        phase=13,
        phase13_aggregator1_type='GeM',
        phase13_aggregator1_cfgs=dict(
            p=3,
            eps=1e-6,
            ),
        phase13_aggregator2_type='GeM',
        phase13_aggregator2_cfgs=dict(
            p=3,
            eps=1e-6,
            ),
        multi_layer=5,
        phase13_feature_train_type='split',
        phase4_overlap_matrix_fuse_type="min",
        phase13_local_correspondence_one_layer=True,
        phase13_min_img_num_pt=[104, 6, 2],
        phase13_min_pc_num_pt=[320, 20, 5],
        phase13_topk=[5000, 5000, 500],
    ),
)

dataset_cfgs=dict(
    train_pc_aug_mode=8,
    eval_pc_aug_mode=3,
    train_image_aug_mode=16,
    eval_image_aug_mode=17,
    raw_dir_name='Boreas_minuse',
    pc_dir_name="Boreas_minuse_40960_to_4096_cliped_fov",
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
    use_image=True,
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
    use_original_pc_correspondence=True,
    correspondence_point_project_threshold=110.0,
)

sampler_cfgs=dict(
    sampler_type='none',
)

batch_sampler_cfgs=dict(
    batch_sampler_type='ensure_k_levels_pos_uniform_multi_v8',
    num_k=3,
    batch_size=16,
    eval_batch_size=16,
    interval_num=50,
    useout_times=1,
)

collect_fn_cfgs=dict(
    collect_fn_type='4',
)

dataloader_cfgs=dict(
    num_workers=4,
    dataset_cfgs=dataset_cfgs,
    sampler_cfgs=sampler_cfgs,
    batch_sampler_cfgs=batch_sampler_cfgs,
    collect_fn_cfgs=collect_fn_cfgs,
)


# loss_cfgs=dict(
#     loss_type='GL_v2_triplet_circle',
#     global_loss_cfgs=dict(
#         base_margin=0.6,
#         normalize_embeddings=False,
#         positive_overlap_ratio=0.2,
#         negative_overlap_ratio=0.05,
#         delta_overlap_ratio=0.01,
#         tuple_formtype='relative_delta',
#         choose_nonzero=True,
#     ),
#     local_loss_cfgs=dict(
#         m=0.2,
#         gamma=40,
#     ),
#     global_loss_weight=0.5,
#     local_loss_weight=0.5,
#     local_positive_mask_margin=[0.5, 0.3, 0.3],
#     local_negative_mask_margin=[0.2, 0.1, 0.2],
# )

loss_cfgs=dict(
    loss_type='GL_double_v2_triplet_circle',
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
        m=0.2,
        gamma=40,
    ),
    global_loss1_weight=0.33,
    global_loss2_weight=0.33,
    local_loss_weight=0.33,
    local_positive_mask_margin=0.3,
    local_negative_mask_margin=0.2,
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

dgcnn_path="cx7jkf53/epoch_73.pth"
resnet_path="thk0qhd5/epoch_73.pth"
# dgcnn_path=None
cmvpr_path=None
pretrained_cfgs=dict(
    cmvprnet=dict(
        image_path=resnet_path,
        image_cfgs=dict(
            backbone_part=2,
        ),
        image_aggregator_cfgs=dict(
            aggregate_part=0,
        ),
        pc_path=dgcnn_path,
        pc_cfgs=dict(
            backbone_part=3,
        ),
        pc_aggregator_cfgs=dict(
            aggregate_part=0,
        ),
        cmvpr_path=cmvpr_path,
    )
)

freeze_cfgs=dict(
    freeze_cmvpr=dict(
        freeze_img_encoder=dict(
            backbone=0,
        ),
        freeze_img_aggregator=dict(
            aggregate=0,
        ),
        freeze_pc_encoder=dict(
            backbone=2,
        ),
        freeze_pc_aggregator=dict(
            aggregate=0,
        ),
    )
)
use_mp=True
need_eval=True














