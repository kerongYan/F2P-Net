import torch
import torch.nn as nn
import numpy as np

class FeatureMSELoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight

    def forward(self, input):
        feats_recon = input["feats_recon"]
        if "gt_block_feats" in input:
            gt_block_feats = input["gt_block_feats"]
            losses = [self.criterion_mse(feats_recon[key], gt_block_feats[key]) for key in feats_recon]
            return torch.sum(torch.stack(losses))
        return torch.tensor(0.0, device=input['image'].device)


class SegmentCrossEntropyLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.weight = weight

    def forward(self, input):
        gt_mask = input["mask"]
        logit = input["logit"]
        logit = logit.view(logit.size(0), 2, -1).permute(0, 2, 1).contiguous().view(-1, 2)
        gt_mask = gt_mask.view(-1).long()
        return self.criterion(logit, gt_mask)


class SegmentFocalLoss(nn.Module):
    def __init__(self, weight=1.0, alpha=None, gamma=2, balance_index=0, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.weight = weight

    def forward(self, input):
        target = input['mask']
        logit = torch.softmax(input['logit'], dim=1)

        num_class = logit.shape[1]
        logit = logit.view(logit.size(0), logit.size(1), -1).permute(0, 2, 1).contiguous().view(-1, num_class)
        target = target.squeeze(1).view(-1, 1)

        # Calculate alpha for class balancing
        alpha = self.alpha if self.alpha is not None else torch.ones(num_class, 1)
        if isinstance(alpha, (list, np.ndarray)):
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1) * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha
        alpha = alpha.to(logit.device)

        idx = target.long()
        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_().scatter_(1, idx, 1).to(logit.device)
        if self.smooth:
            one_hot_key = torch.clamp(one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)

        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        loss = -alpha[idx] * torch.pow((1 - pt), self.gamma) * logpt
        return loss.mean() if self.weight else loss.sum()


class ImageMSELoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.criterion_mse = nn.MSELoss()
        self.weight = weight

    def forward(self, input):
        return self.criterion_mse(input["ori"], input["recon"])


class ClassifierCrossEntropyLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.weight = weight

    def forward(self, input):
        return self.criterion(input["pred"], input["label"].long())

def get_loss_functions():
    loss_functions = {
        "SegmentCrossEntropyLoss": SegmentCrossEntropyLoss(weight=1.0),
        "FeatureMSELoss": FeatureMSELoss(weight=1.0),
        "SegmentFocalLoss": SegmentFocalLoss(weight=1.0, alpha=0.25, gamma=2, balance_index=0, smooth=1e-5),
        "ImageMSELoss": ImageMSELoss(weight=1.0),
        "ClassifierCrossEntropyLoss": ClassifierCrossEntropyLoss(weight=1.0),
    }
    return loss_functions

