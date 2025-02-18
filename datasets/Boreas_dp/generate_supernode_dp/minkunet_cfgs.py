model_cfgs = dict(IGNORE_LABEL=0,
IN_FEATURE_DIM=4,
BLOCK='ResBlock',
NUM_LAYER=[2, 3, 4, 6, 2, 2, 2, 2],
PLANES=[32, 32, 64, 128, 256, 256, 128, 96, 96],
cr=1.6,
DROPOUT_P=0.0,
IF_DIST=False,)
