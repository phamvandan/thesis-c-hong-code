import torch
import torchvision
import torch.nn as nn


class Spatial_Loss(nn.Module):
    def __init__(self, in_channels):
        super(Spatial_Loss, self).__init__()
        self.res_scale = in_channels
        
        self.make_PAN = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

        self.L1_loss = nn.L1Loss()
        
    def forward(self, pred_HS, ref_HS):
        pan_pred = self.make_PAN(pred_HS)
        with torch.no_grad():
            pan_ref = self.make_PAN(ref_HS)     
        spatial_loss = self.L1_loss(pan_pred, pan_ref.detach())
        return spatial_loss

def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
    Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)