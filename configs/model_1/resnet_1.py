
_base_ = ['../base/models/resnet.py',
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
# resnet_1_path=None
resnet_1_path='/home/pengjianyi/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth'
pretrained_cfgs=dict(
    imgnet=dict(
        path=resnet_1_path,
    )
)