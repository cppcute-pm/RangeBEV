
loss_cfgs=dict(
    loss_type='infonce',
    temperature=1.0, # 0.1, 0.01, 0.03, 1.0 is from LIP-Loc
    reduction='mean',
    negative_mode=1, # TODO: 2、3
    positive_mode=1, # TODO: 2、3
    distance_mode=2,
)