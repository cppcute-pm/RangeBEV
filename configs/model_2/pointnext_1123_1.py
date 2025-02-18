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
        backbone_type='PointNext',
        out_layer=1,
        backbone_config=dict(
            in_channels=4,
            width=32,
            radius=3.5,
            nsample=32,
            dropout=0.5,
            f_layer=9, # 4096
            c_layer=5, # 16
            pointnextv3_op=1,
        ),
        aggregate_type='GeM',
        aggregate_config=dict(
            p=3,
            eps=1e-6,
        ),
    ),
)

dataset_cfgs=dict(
    data_type='kitti',
    train_range_aug_mode=17,
    eval_range_aug_mode=17,
    train_image_aug_mode=16,
    eval_image_aug_mode=17,
    train_pc_aug_mode=8,
    eval_pc_aug_mode=3,
    kitti_data_root='KITTI',
    semantickitti_data_root='semanticKITTI',
    raw_dir_name='dataset',
    pc_dir_name='16384_to_4096_cliped_fov',
    image_dir_name='768x128_image',
    range_dir_name='16384_to_4096_cliped_fov_range_image',
    train_coords_filename='train_UTM_coords.pkl',
    test_coords_filename='test_UTM_coords.pkl',
    image_size=[128, 768],
    range_img_size=[64, 224],
    use_cloud=True,
    use_image=False,
    use_range=False,
    num_points=4096,
    use_overlap_ratio=False,
    crop_location=None,
    dist_caculation_type='all_coords_L2',
    overlap_ratio_type='pose_dist_sim',
    pose_dist_threshold=7.5,
    train_sequence_list=[
        '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'
    ],
    test_sequence_list=[
        '00', '02', '05', '06', '07', '08'
    ],
    true_neighbour_dist=10.0,
    positive_distance=10.0,
    non_negative_distance=10.0,
    use_original_pc_correspondence=False,
)

batch_sampler_cfgs = dict(
    batch_sampler_type='ensure_k_levels_pos_uniform_multi_v19',
    num_k=3,
    batch_size=16,
    eval_batch_size=16,
    interval_num=5,
    useout_times=1)

sampler_cfgs=dict(
    sampler_type='none',
)

collect_fn_cfgs=dict(
    collect_fn_type='5',
)

dataloader_cfgs=dict(
    batch_sampler_cfgs=batch_sampler_cfgs,
    num_workers=4,
    dataset_cfgs=dataset_cfgs,
    sampler_cfgs=sampler_cfgs,
    collect_fn_cfgs=collect_fn_cfgs,
)

loss_cfgs = dict(
    loss_type='triplet',
    margin=0.2,
    normalize_embeddings=False,
    hard_mining=True,
    pair_dist_info=True)


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
        backbone=4,
        aggregate=0,
        )
)

evaluate_normalize=False
evaluator_type=5