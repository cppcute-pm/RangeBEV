# only for the images
start_epoch=0
epoch=80
save_interval=1
val_interval=3
train_val=False
need_eval=True
eval_interval=2
evaluator_type=1
use_mp=False
accumulate_iter=1
find_unused_parameters=False
all_gather_cfgs=dict(
    all_gather_flag=False,
)