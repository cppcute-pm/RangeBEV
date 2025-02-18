_base_ = ['../base/models/dgcnn.py',
            '../base/datasets/oxford.py',
            '../base/losses/infonce.py',
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

loss_cfgs=dict(
    loss_type='infonce',
    temperature=0.001, 
    reduction='mean',
    negative_mode=2,
    positive_mode=3,
    distance_mode=2,
)

dataset_cfgs=dict(
    use_cloud=True,
    use_image=False,
)
dataloader_cfgs=dict(
    dataset_cfgs=dataset_cfgs,
)
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