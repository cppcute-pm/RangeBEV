
_base_ = ['../base/models/ResFPN.py',
            '../base/datasets/oxford.py',
            '../base/losses/triplet.py',
            '../base/optimizers/adam.py',
            '../base/schedulers/CosineAnnealingWarmRestarts.py',
            '../base/runtime.py',]
dataset_cfgs=dict(
    use_cloud=False,
    use_image=True,
)
dataloader_cfgs=dict(
    dataset_cfgs=dataset_cfgs,
)
resfpn_1_path=None
pretrained_cfgs=dict(
    imgnet=dict(
        path=resfpn_1_path,
    )
)