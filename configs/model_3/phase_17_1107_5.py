dataset_cfgs = dict(
    data_type='kitti',
    train_pc_aug_mode=8,
    eval_pc_aug_mode=3,
    train_image_aug_mode=16,
    eval_image_aug_mode=17,
    kitti_data_root='KITTI',
    semantickitti_data_root='semanticKITTI',
    raw_dir_name='dataset',
    pc_dir_name='16384_to_4096_cliped_fov',
    image_dir_name="768x128_image",
    train_coords_filename='train_UTM_coords_v1_17m.pkl',
    test_coords_filename='test_UTM_coords.pkl',
    image_size=[128, 768],
    use_cloud=True,
    use_image=True,
    num_points=4096,
    use_overlap_ratio=True,
    crop_location=None,
    dist_caculation_type='all_coords_L2_mean',
    overlap_ratio_type='pose_dist_sim',
    pose_dist_threshold=12.75,
    train_sequence_list=['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'],
    test_sequence_list=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'],
    true_neighbour_dist=10.0,
    use_original_pc_correspondence=False,
    correspondence_point_project_threshold=110.0,
    # below parameters are only for program running
    positive_distance=10.0,
    non_negative_distance=20.0,)
sampler_cfgs = dict(sampler_type='none')
batch_sampler_cfgs = dict(
    batch_sampler_type='ensure_k_levels_pos_uniform_multi_v19',
    num_k=3,
    batch_size=16,
    eval_batch_size=16,
    interval_num=5,
    useout_times=1)
collect_fn_cfgs = dict(collect_fn_type='5')
dataloader_cfgs = dict(
    num_workers=4,
    dataset_cfgs=dataset_cfgs,
    sampler_cfgs=sampler_cfgs,
    batch_sampler_cfgs=batch_sampler_cfgs,
    collect_fn_cfgs=collect_fn_cfgs)
optimizer_cfgs = dict(optimizer_type='adam', lr=1e-05, weight_decay=0.0001)
scheduler_cfgs = dict(
    scheduler_type='CosineAnnealingWarmRestarts',
    warmup_epoch=5,
    min_lr=1e-07,
    T_mult=2)
start_epoch = 0
epoch = 80
save_interval = 1
val_interval = 3
train_val = False
need_eval = True
eval_interval = 2
evaluator_type = 5
use_mp = True
accumulate_iter = 1
find_unused_parameters = False
all_gather_cfgs = dict(all_gather_flag=False)
model_cfgs = dict(
    modal_type=3,
    out_dim=256,
    cmvpr_cfgs=dict(
        image_encoder_type='ResUNetmmseg',
        image_out_layer=None,
        pc_out_layer=None,
        image_encoder_out_layer=3,
        image_encoder_cfgs=dict(
            f_layer=3,
            decoder_head_1=dict(
                block_type='Bottleneck_exp_rate_2',
                stage_blocks=[3, 6, 4, 3],
                channels_list=[2048, 1024, 512, 256, 64],
                upsample_cfg=dict(type='InterpConv'),
                last_layer_num_blocks=2)),
        pc_encoder_type='PointNext',
        pc_encoder_out_layer=3,
        pc_encoder_cfgs=dict(
            in_channels=4,
            width=32,
            radius=3.5,
            nsample=32,
            dropout=0.5,
            pointnextv3_op=1,
            f_layer=6,
            c_layer=5),
        image_aggregator_type='GeM',
        image_aggregator_cfgs=dict(p=3, eps=1e-06),
        pc_aggregator_type='GeM',
        pc_aggregator_cfgs=dict(p=3, eps=1e-06),
        phase=17,
        phase17_use_four_aggregator=False,
        phase17_use_SA_block=False,
        phase17_aggregator_type='GeM',
        phase17_aggregator_cfgs=dict(p=3, eps=1e-06)))
loss_cfgs = dict(
    loss_type='v2_triplet_cm',
    base_margin=0.6,
    normalize_embeddings=False,
    positive_overlap_ratio=0.2,
    negative_overlap_ratio=0.05,
    delta_overlap_ratio=0.01,
    tuple_formtype='relative_delta',
    choose_nonzero=True,)
dgcnn_path = None
resnet_path = "/home/pengjianyi/.cache/torch/hub/checkpoints/fpn_r50_512x512_160k_ade20k_20200718_131734-5b5a6ab9_v2.pth"
cmvpr_path = None
pretrained_cfgs = dict(
    cmvprnet=dict(
        image_path=resnet_path,
        image_cfgs=dict(backbone_part=5),
        image_aggregator_cfgs=dict(aggregate_part=0),
        pc_path=dgcnn_path,
        pc_cfgs=dict(backbone_part=2),
        pc_aggregator_cfgs=dict(aggregate_part=0),
        cmvpr_path=None))
freeze_cfgs = dict(
    freeze_cmvpr=dict(
        freeze_img_encoder=dict(backbone=3),
        freeze_img_aggregator=dict(aggregate=0),
        freeze_pc_encoder=dict(backbone=7),
        freeze_pc_aggregator=dict(aggregate=0)))
