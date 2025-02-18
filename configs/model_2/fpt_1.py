_base_ = ['../base/models/fpt.py',
            '../base/datasets/oxford.py',
            '../base/losses/triplet.py',
            '../base/optimizers/adam.py',
            '../base/schedulers/CosineAnnealingWarmRestarts.py',
            '../base/runtime.py',]

model_cfgs=dict(
    pcnet_cfgs=dict(
        out_layer=2,
    ),
)


dataset_cfgs=dict(
    use_cloud=True,
    use_image=False,
)
dataloader_cfgs=dict(
    dataset_cfgs=dataset_cfgs,
)
fpt_1_path="/home/pengjianyi/.cache/torch/hub/checkpoints/fpt_scannet_2cm_v2.ckpt"
# fpt_1_path=None
pretrained_cfgs=dict(
    pcnet=dict(
        path=fpt_1_path,
        backbone_part=0,
    )
)

freeze_cfgs=dict(
    freeze_pcnet=dict(
        backbone=2, # TODO: backbone=2
        )
)