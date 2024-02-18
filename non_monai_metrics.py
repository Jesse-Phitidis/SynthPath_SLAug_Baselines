import isles_scoring
from sklearn.metrics import average_precision_score
from typing import Union
import torch
import numpy as np
from functools import partial


class NonMONAIMetrics:
    
    def __init__(self, include_background=True, reduction="none"):
        if reduction not in ["none", "mean", "sum"]:
            raise NotImplementedError
        
        self.reduction = reduction
        self.include_background = include_background
        self.pre, self.rec, self.lf1, self.lpre, self.lrec, self.ap = [], [], [], [], [], []
        
    def __call__(self, pred: torch.tensor, gt: torch.tensor, pred_soft: Union[torch.tensor, None] = None):
            
        pred = pred.detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()
        if pred_soft is not None:
            pred_soft = pred_soft.detach().cpu().numpy()
        
        n_batch = gt.shape[0]
        if n_batch > 1:
            raise NotImplementedError
        n_classes = gt.shape[1]
        assert not ((not self.include_background) and (n_classes==1)), "cannot set include_background to false for single channel predictions"
        
        if not self.include_background:
            pred = pred[1:, ...]
            gt = gt[1:, ...]
        
        pre, rec, lf1, lpre, lrec, ap = [], [], [], [], [], []
        for i in range(n_classes):
            i_pred = pred[0, i]
            i_gt = gt[0, i]
            if pred_soft is not None:
                i_pred_soft = pred_soft[0, i]
            
            pre.append(isles_scoring.precision(i_gt, i_pred))
            rec.append(isles_scoring.sensitivity(i_gt, i_pred))
            _lf1, _lpre, _lrec = isles_scoring._lesion_metrics_score(i_gt, i_pred)
            lf1.append(_lf1)
            lpre.append(_lpre)
            lrec.append(_lrec)
            if pred_soft is not None:
                ap.append(average_precision_score(i_gt.flatten().astype(int), i_pred_soft.flatten()))
        
        self.pre.append(pre), self.rec.append(rec), self.lf1.append(lf1), self.lpre.append(lpre), self.lrec.append(lrec)
        if pred_soft is not None:
            self.ap.append(ap)
            return torch.tensor(pre).unsqueeze(0), torch.tensor(rec).unsqueeze(0), torch.tensor(lf1).unsqueeze(0), \
                   torch.tensor(lpre).unsqueeze(0), torch.tensor(lrec).unsqueeze(0), torch.tensor(ap).unsqueeze(0)
                   
        return torch.tensor(pre).unsqueeze(0), torch.tensor(rec).unsqueeze(0), torch.tensor(lf1).unsqueeze(0), \
                   torch.tensor(lpre).unsqueeze(0), torch.tensor(lrec).unsqueeze(0)
    
    def aggregate(self):
        
        if self.reduction == "mean":
            reduce = partial(torch.nanmean, dim=0)
        if self.reduction == "sum":
            reduce = partial(torch.nansum, dim=0)
        if self.reduction == "none":
            reduce = lambda x: x
            
        pre = reduce(torch.from_numpy(np.stack(self.pre, axis=0)))
        rec = reduce(torch.from_numpy(np.stack(self.rec, axis=0)))
        lf1 = reduce(torch.from_numpy(np.stack(self.lf1, axis=0)))
        lpre = reduce(torch.from_numpy(np.stack(self.lpre, axis=0)))
        lrec = reduce(torch.from_numpy(np.stack(self.lrec, axis=0)))
        
        if len(self.ap) > 0:
            ap = reduce(torch.from_numpy(np.stack(self.ap, axis=0)))
            return pre, rec, lf1, lpre, lrec, ap
        return pre, rec, lf1, lpre, lrec
    
    def reset(self):
        del self.pre, self.rec, self.lf1, self.lpre, self.lrec, self.ap
        self.pre, self.rec, self.lf1, self.lpre, self.lrec, self.ap = [], [], [], [], [], []