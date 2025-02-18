_base_ = ['../base/datasets/boreas.py',
            '../base/losses/triplet.py',
            '../base/optimizers/adam.py',
            '../base/schedulers/CosineAnnealingWarmRestarts.py',
            '../base/runtime.py',]
model_cfgs=dict(
    modal_type=3,
    out_dim=256,
    cmvpr_cfgs=dict(
        image_encoder_type='CCT',
        image_out_layer=2,
        pc_out_layer=1,
        image_encoder_out_layer=3,
        image_encoder_cfgs=dict(
            f_layer=8,
            c_layer=8,
            img_size=224,
            ),
        pc_encoder_type='DGCNN',
        pc_encoder_out_layer=3,
        pc_encoder_cfgs=dict(
            out_k=20,
            k=20,
            emb_dims=256,
            dropout=0.5,
            f_layer=2,
            c_layer=5,
            reducer=[2, 2, 2, 2, 2]
            ),
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
        phase=1,
    ),
)

dataset_cfgs=dict(
    train_pc_aug_mode=5,
    eval_pc_aug_mode=0,
    train_image_aug_mode=1,
    eval_image_aug_mode=3,
    raw_dir_name='Boreas_minuse',
    pc_dir_name="Boreas_minuse_163840_to_4096",
    rendered_dir_name=None,
    image_dir_name=None,
    mask_dir_name=None,
    tool_name='my_tool',
    train_query_filename='class_split_train_queries.pickle',
    val_query_filename='class_split_test_queries.pickle',
    eval_query_filename='class_split_test_queries.pickle',
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
    collect_fn_type='2',
)

dataloader_cfgs=dict(
    batch_sampler_cfgs=batch_sampler_cfgs,
    num_workers=8,
    dataset_cfgs=dataset_cfgs,
    sampler_cfgs=sampler_cfgs,
    collect_fn_cfgs=collect_fn_cfgs,
)

cct_path="rd3q6zdk/epoch_73.pth"
dgcnn_path="4jhsl2hh/epoch_69.pth"
# cct_path=None
# dgcnn_path=None

loss_cfgs=dict(
    loss_type='triplet_cm',
    pair_dist_info=True,
    hard_mining=True,
    margin=0.2,
    normalize_embeddings=False,
)

pretrained_cfgs=dict(
    cmvprnet=dict(
        image_path=cct_path,
        image_cfgs=dict(
            backbone_part=0
        ),
        image_aggregator_cfgs=dict(
            aggregate_part=0
        ),
        pc_path=dgcnn_path,
        pc_cfgs=dict(
            backbone_part=0
        ),
        pc_aggregator_cfgs=dict(
            aggregate_part=0
        ),
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
            backbone=0,
        ),
        freeze_pc_aggregator=dict(
            aggregate=0,
        ),
    )
)

use_mp=True
# find_unused_parameters=False
# all_gather_cfgs=dict(
#     all_gather_flag=True,
#     types=['embeddings1', 'embeddings2', 'positives_mask', 'negatives_mask']
# )
# accumulate_iter=8 
