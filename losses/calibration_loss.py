import torch
from pytorch_metric_learning import losses, distances
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


class CalibrationLoss(losses.ContrastiveLoss):

    def __init__(self, cfgs):
        super().__init__(pos_margin=cfgs.pos_margin, neg_margin=cfgs.neg_margin)

    def get_default_distance(self):
        return distances.DotProductSimilarity()

    def forward(self, embeddings, positives_mask, negatives_mask):
        triplet_mask = torch.logical_and(positives_mask.unsqueeze(-1), negatives_mask.unsqueeze(-2))
        indices_tuple = torch.nonzero(triplet_mask, as_tuple=True)
        return super().forward(embeddings, indices_tuple=indices_tuple)