_base_ = ['../base/models/pointmlp.py',
            '../base/datasets/oxford.py',
            '../base/losses/triplet.py',
            '../base/optimizers/adam.py',
            '../base/schedulers/CosineAnnealingWarmRestarts.py',
            '../base/runtime.py',]

model_cfgs=dict(
    pcnet_cfgs=dict(
        backbone_config=dict(
            num_points=1024,
            ),
    ),
)

dataset_cfgs=dict(
    use_cloud=True,
    use_image=False,
)
dataloader_cfgs=dict(
    dataset_cfgs=dataset_cfgs,
)
pointmlp_1_path="/home/pengjianyi/.cache/torch/hub/checkpoints/pointmlp_scanNN_best_checkpoint_v2.pth"
# pointmlp_1_path=None
pretrained_cfgs=dict(
    pcnet=dict(
        path=pointmlp_1_path,
        backbone_part=0,
    )
)

freeze_cfgs=dict(
    freeze_pcnet=dict(
        backbone=2,
        )
)