import torch
import torch.nn as nn
from monai.losses import DiceLoss

class SetCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss=nn.CrossEntropyLoss()
        self.dice_loss=DiceLoss(to_onehot_y=True,softmax=True,squared_pred=True,smooth_nr=0.0,smooth_dr=1e-6)
        self.weight_dict={'ce_loss':1, 'dice_loss':1}

    def get_loss(self,  pred, gt):
        
        # Remove channels except 0 and 19 so we now do softmax over background and stroke
        pred = pred[:, (0,19), ...]
        # Remove gt labels for anatomy and keep only stroke
        gt = torch.where(gt==19, 1, 0)
        
        if len(gt.size())==4 and gt.size(1)==1:
            gt=gt[:,0]

        if type(pred) is not list:
            _ce=self.ce_loss(pred,gt)
            _dc=self.dice_loss(pred,gt.unsqueeze(1))
            return {'ce_loss': _ce,'dice_loss':_dc}
        else:
            ce=0
            dc=0
            for p in pred:
                ce+=self.ce_loss(p,gt)
                dc+=self.dice_loss(p,gt.unsqueeze(1))
            return {'ce_loss': ce, 'dice_loss':dc}
