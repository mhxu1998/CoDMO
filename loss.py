import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    # @torchsnooper.snoop()
    def forward(self, y_hat, y, length):
        # Calculate the loss
        logEps = torch.tensor(1e-8)
        tmp1 = y * torch.log(y_hat + logEps)
        tmp2 = (1. - y) * torch.log(1. - y_hat + logEps)
        cross_entropy = -(tmp1 + tmp2)
        tmp = torch.sum(cross_entropy, 2)
        tmp = torch.sum(tmp, 0)
        log_likelyhood = torch.div(tmp, length)
        cost = torch.mean(log_likelyhood)

        # Get the top k results
        y_sorted, indices = torch.sort(y_hat, dim=2, descending=True)

        TP_5 = 0.
        TN_5 = 0.
        indices_5 = indices[:, :, :5]
        for i, j, k in torch.nonzero(y, as_tuple=False):
            if k in indices_5[i][j]:
                TP_5 += 1
            else:
                TN_5 += 1

        TP_10 = 0.
        TN_10 = 0.
        indices_10 = indices[:, :, :10]
        for i, j, k in torch.nonzero(y, as_tuple=False):
            if k in indices_10[i][j]:
                TP_10 += 1
            else:
                TN_10 += 1
        acc = TP_10 / (TP_10 + TN_10)

        TP_15 = 0.
        TN_15 = 0.
        indices_15 = indices[:, :, :15]
        for i, j, k in torch.nonzero(y, as_tuple=False):
            if k in indices_15[i][j]:
                TP_15 += 1
            else:
                TN_15 += 1

        TP_20 = 0.
        TN_20 = 0.
        indices_20 = indices[:, :, :20]
        for i, j, k in torch.nonzero(y, as_tuple=False):
            if k in indices_20[i][j]:
                TP_20 += 1
            else:
                TN_20 += 1

        return cost, [TP_5, TP_10, TP_15, TP_20], [TP_5+TN_5, TP_10+TN_10, TP_15+TN_15, TP_20+TN_20]
