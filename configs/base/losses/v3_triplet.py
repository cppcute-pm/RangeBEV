loss_cfgs=dict(
    loss_type='v3_triplet',
    base_margin=2.0,
    normalize_embeddings=False,
    positive_overlap_ratio=0.2,
    negative_overlap_ratio=0.01,
    delta_overlap_ratio=0.01,
    tuple_formtype='relative_delta',
)