import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temp=0.1, q=0.7, GSC=False):
        super(SupConLoss, self).__init__()
        self.temp = temp
        self.q = q
        self.GSC = GSC

    def forward(self, x, batch_label):
        indexes = batch_label.unsqueeze(0)
        same_label = (indexes.t() == indexes) # 각 샘플과 같은 레이블을 갖는 애들 true -> diagonal mat
        label_mask = same_label.to(int).to(x.device)
        sim_proj = torch.einsum("if, jf -> ij", x, x) # torch.mm(x, x.t())
        sim = sim_proj / self.temp # z{i,t}*z{k}/T -> 배치안에 샘플 모두 서로 유사도 구함
        # for numerical stability
        logits_max, _ = torch.max(sim, dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        logits_mask = torch.scatter(
            torch.ones_like(sim).to(x.device),
            1,
            torch.arange(sim.size(-1)).view(-1, 1).to(x.device),
            0
        ).to(x.device)

        mask = label_mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        if self.GSC:
            py = log_prob.clone().detach()
            py = torch.exp(py)
            loss_weight = (py**self.q)*self.q
            log_prob = loss_weight * log_prob

        # compute mean of log-likelihood over positive
        sum_mask = mask.sum(1)
        sum_mask[sum_mask == 0] = 1
        mean_log_prob_pos = (mask * log_prob).sum(1) / sum_mask

        # loss
        loss = -mean_log_prob_pos
        loss = loss.mean()
        return loss