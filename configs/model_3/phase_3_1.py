_base_ = ['../base/datasets/oxford.py',
            '../base/losses/cmpm.py',
            '../base/optimizers/adam.py',
            '../base/schedulers/CosineAnnealingWarmRestarts.py',
            '../base/runtime.py',]
model_cfgs=dict(
    modal_type=3,
    out_dim=256,
    cmvpr_cfgs=dict(
        image_encoder_type='CCT',
        image_encoder_out_layer=3,
        image_encoder_cfgs=dict(
            f_layer=8,
            c_layer=14,
            img_size=224,
            ),
        pc_encoder_type='DGCNN',
        pc_encoder_out_layer=3,
        pc_encoder_cfgs=dict(
            k=8,
            out_k=8,
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
        phase=3,
        phase_new_PoA=1,
        phase3_proj_type=1,
        phase3_aggregator1_type='NetVLAD',
        phase3_aggregator1_cfgs=dict(
            gating=True,
            add_batch_norm=False,
            cluster_size=64,
            ),
        phase3_aggregator2_type='NetVLAD',
        phase3_aggregator2_cfgs=dict(
            gating=True,
            add_batch_norm=False,
            cluster_size=64,
            ),
    ),
)

dataset_cfgs=dict(
    use_cloud=True,
    use_image=True,
)

dataloader_cfgs=dict(
    batch_sampler_cfgs=dict(
        batch_sampler_type='ensure_k_pos',
        batch_size=16,
        batch_size_limit=16,
        num_k=2,
    ),
    dataset_cfgs=dataset_cfgs,
)
# cct_path="barhr61h/epoch_69.pth"
# dgcnn_path="dwu15crt/epoch_71.pth"
cct_path="tndumke3/epoch_51.pth"
dgcnn_path="eq63384y/epoch_73.pth"
# cct_1_path=None
loss_cfgs=dict(
    loss_type='cmpm',
    avg_sim_info=True,
    epsilon=1e-6,
    SDM_logit_scale=50, # like the IRRA
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