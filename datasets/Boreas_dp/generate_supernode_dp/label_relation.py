cityscapes_label=list(
                  'road',           # 0
                  'sidewwalk',      # 1
                  'building',       # 2
                  'wall',           # 3
                  'fence',          # 4
                  'pole',           # 5
                  'traffic light',  # 6
                  'traffic sign',   # 7
                  'vegetation',     # 8
                  'terrain',        # 9
                  'sky',            # 10
                  'person',         # 11
                  'rider',          # 12
                  'car',            # 13
                  'truck',          # 14
                  'bus',            # 15
                  'train',          # 16
                  'motorcycle',     # 17
                  'bicycle',        # 18
                )
cityscapes_ignored_label=list(
                  10,
                  11,
                  12,
                  13,
                  14,
                  15,
                  16,
                  17,
                  18,
                )

semanticKitti_label=list(
    'unlabeled',   # 0
    'car',         # 1
    'bicycle',     # 2
    'motorcycle',  # 3
    'truck',       # 4
    'other-vehicle', # 5
    'person',      # 6
    'bicyclist',   # 7
    'motorcyclist',# 8
    'road',        # 9
    'parking',     # 10
    'sidewalk',    # 11
    'other-ground',# 12
    'building',    # 13
    'fence',       # 14
    'vegetation',  # 15
    'trunk',      # 16
    'terrian',     # 17
    'pole',        # 18
    'traffic-sign',# 19
)

semanticKitti_ignored_label=list(
  # 0 should be carefully considered because of the snow noise and other things
                  1,
                  2,
                  3,
                  4,
                  5,
                  6,
                  7,
                  8,
                )

cityscapes_label_in_semanticKitti_label = list(
    list(9, 9, 9, 9),
    list(11, 11, 11, 11),
    list(13, 13, 13, 13),
    list(13, 13, 0, 0),
    list(14, 14, 14, 14),
    list(18, 18, 18, 18),
    list(19, 19, 0, 0),
    list(19, 19, 19, 19),
    list(15, 15, 16, 16),
    list(17, 17, 12, 12),
    list(-1, -1, -1, -1),
    list(6, 6, 6, 6),
    list(7, 7, 8, 8),
    list(1, 1, 1, 1),
    list(4, 4, 4, 4),
    list(5, 1, 4, 4),
    list(0, 1, 4, 5),
    list(3, 3, 3, 3),
    list(2, 2, 2, 2),
)