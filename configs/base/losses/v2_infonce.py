loss_cfgs=dict(
    loss_type='v2_infonce',
    temperature=0.001, 
    reduction='mean',
    negative_mode=1, # TODO: 2、3
    positive_mode=1, # TODO: 2、3
    distance_mode=2,
)