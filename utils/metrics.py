import torch
import numpy as np


class Metrics:

    def __init__(self, top_k_metrics):
        self.top_k_metrics = top_k_metrics

    def calculate_hitratio(self, predictions, target):
        # calculate HitRatio@k metric
        _, topk_idx = predictions.topk(self.top_k_metrics, dim=-1)
        relevance = target.take_along_dim(topk_idx, dim=-1)

        return (relevance.sum(dim=-1) > 0).float().mean().item()

    def dcg_metric(self, target):
        batch_size, k = target.shape
        rank_positions = torch.arange(1, k + 1, dtype=torch.float32, device=target.device).tile((batch_size, 1))
        
        return (target / torch.log2(rank_positions + 1)).sum(dim=-1)

    def calculate_ndcg(self, predictions, target):
        # calculate NDCG@k metric
        _, topk_idx = predictions.topk(self.top_k_metrics, dim=-1)
        relevance = target.take_along_dim(topk_idx, dim=-1)
        ideal_target, _ = target.topk(self.top_k_metrics, dim=-1)

        return (self.dcg_metric(relevance) / self.dcg_metric(ideal_target)).mean().item()

    def calculate_mrr(self, predictions, target):
        # calculate MRR metric
        sorted_indices = predictions.sort(descending=True).indices
        relevance = target.take_along_dim(sorted_indices, dim=-1)
        first_relevant_positions = relevance.argmax(dim=-1) + 1
        
        return (1. / first_relevant_positions).mean().item()

    def calculate_all(self, predictions, y_true, num_classes):
        target = torch.nn.functional.one_hot(y_true, num_classes=num_classes)

        hitratio = self.calculate_hitratio(predictions, target)
        ndcg = self.calculate_ndcg(predictions, target)
        mrr = self.calculate_mrr(predictions, target)

        return hitratio, ndcg, mrr
    
    def calculate_all_with_target(self, predictions, target):
        hitratio = self.calculate_hitratio(predictions, target)
        ndcg = self.calculate_ndcg(predictions, target)
        mrr = self.calculate_mrr(predictions, target)

        return hitratio, ndcg, mrr