
_base_ = [
            '../base/datasets/boreas_v2.py',
            '../base/losses/v2_triplet.py',
            '../base/optimizers/adam.py',
            '../base/schedulers/CosineAnnealingWarmRestarts.py',
            '../base/runtime.py',]
dataset_cfgs=dict(
    data_type='kitti',
    train_image_aug_mode=16,
    eval_image_aug_mode=17,
    train_pc_aug_mode=0,
    eval_pc_aug_mode=0,
    kitti_data_root='KITTI',
    semantickitti_data_root='semanticKITTI',
    raw_dir_name='dataset',
    pc_dir_name=None,
    image_dir_name='768x128_image',
    range_dir_name=None,
    train_coords_filename='train_UTM_coords.pkl',
    test_coords_filename='test_UTM_coords.pkl',
    image_size=[128, 768],
    range_img_size=None,
    use_cloud=False,
    use_image=True,
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

sampler_cfgs=dict(
    sampler_type='none',
)

batch_sampler_cfgs=dict(
    batch_sampler_type='ensure_k_levels_pos_uniform_multi_v19',
    num_k=3,
    batch_size=16,
    eval_batch_size=16,
    interval_num=5,
    useout_times=1,
)

collect_fn_cfgs=dict(
    collect_fn_type='5',
)

dataloader_cfgs=dict(
    num_workers=4,
    dataset_cfgs=dataset_cfgs,
    sampler_cfgs=sampler_cfgs,
    batch_sampler_cfgs=batch_sampler_cfgs,
    collect_fn_cfgs=collect_fn_cfgs,
)

loss_cfgs = dict(
    loss_type='triplet',
    margin=0.2,
    normalize_embeddings=False,
    hard_mining=True,
    pair_dist_info=True)

model_cfgs=dict(
    modal_type=1, # '1' '2' '3'
    out_dim=256,
    imgnet_cfgs=dict(
        backbone_type='ResFPNmmseg',
        out_layer=1,
        backbone_config=dict(
            f_layer=8, # 56 * 56
            c_layer=4, # 7 * 7
            norm_cfg=dict(type='BN', requires_grad=True),
            backbone=dict(
                depth=50,
                in_channels=3,
                num_stages=4,
                strides=(1, 2, 2, 2),
                dilations=(1, 1, 1, 1),
                out_indices=(0, 1, 2, 3),
                norm_eval=False,
                style='pytorch',
                contract_dilation=True,
            ),
            neck=dict(
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                num_outs=4,
            ),
            decode_head=dict(
                in_channels=[256, 256, 256, 256],
                in_index=[0, 1, 2, 3],
                feature_strides=[4, 8, 16, 32],
                channels=128,
                dropout_ratio=0.1,
                align_corners=True,
                loss_decode=dict(
                    type='CrossEntropyLoss', 
                    use_sigmoid=False, 
                    loss_weight=1.0,
                ),
            ),
            new_definition=True,
        ),
        aggregate_type='GeM',
        aggregate_config=dict(
            p=3,
            eps=1e-6,
        ),
    ),
    )

# resfpnmmseg_path=None
resfpnmmseg_path="/home/pengjianyi/.cache/torch/hub/checkpoints/fpn_r50_512x512_160k_ade20k_20200718_131734-5b5a6ab9_v2.pth"
freeze_cfgs=dict(
    freeze_imgnet=dict(
        backbone=5,
        aggregate=0,
        )
)
pretrained_cfgs=dict(
    imgnet=dict(
        path=resfpnmmseg_path,
        backbone_part=0,
        aggregate_part=None,
    )
)

use_mp=True
evaluate_normalize=False
evaluator_type=5