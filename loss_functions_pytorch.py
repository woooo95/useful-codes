import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F

#DiceLoss
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        
        smooth = 0
        # m1=pred.flatten()
        # m2=target.flatten()
        # intersection = (m1 * m2)

        # score=1-((2. * torch.sum(intersection) + smooth) / (torch.sum(m1) + torch.sum(m2) + smooth))
        # #score=1-((2. * torch.sum(intersection) + smooth) / (torch.sum(m1*m1) + torch.sum(m2*m2) + smooth))
                
        num = target.shape[0]
        m1 = pred.view(num, -1)
        m2 = target.view(num, -1)
        intersection=torch.mul(m1,m2)
        score = 1-torch.sum((2. * torch.sum(intersection,dim=1) + smooth) / (torch.sum(m1,dim=1) + torch.sum(m2,dim=1) + smooth))/num
        
        # for squared
        ## score = 1-torch.sum((2. * torch.sum(intersection,dim=1) + smooth) / (torch.sum(m1*m1,dim=1) + torch.sum(m2*m2,dim=1) + smooth))/num
        
        return score


#BCE-DiceLoss
class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE


#Jaccard/Intersection over Union (IoU) Loss
class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU


#Focal Loss
Focal_ALPHA = 0.8
Focal_GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=Focal_ALPHA, gamma=Focal_GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss


#Tversky Loss
Tversky_ALPHA = 0.5
Tversky_BETA = 0.5

class TverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(TverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=Tversky_ALPHA, beta=Tversky_BETA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
        return 1 - Tversky


#Focal Tversky Loss
FT_ALPHA = 0.5
FT_BETA = 0.5
FT_GAMMA = 1

class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=FT_ALPHA, beta=FT_BETA, gamma=FT_GAMMA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky



#Lovasz Hinge Loss
class LovaszHingeLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, inputs, targets):
        inputs = F.sigmoid(inputs)    
        Lovasz = lovasz_hinge(inputs, targets, per_image=False)                       
        return Lovasz



#Combo Loss
ALPHA = 0.5 # < 0.5 penalises FP more, > 0.5 penalises FN more
BETA = 0.5 #weighted contribution of modified CE loss compared to Dice loss
e = 1e-5

class ComboLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(ComboLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA):
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        intersection = (inputs * targets).sum()    
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        
        inputs = torch.clamp(inputs, e, 1.0 - e)       
        out = - (BETA * ((targets * torch.log(inputs)) + ((1 - BETA) * (1.0 - targets) * torch.log(1.0 - inputs))))
        weighted_ce = out.mean(-1)
        combo = (ALPHA * weighted_ce) - ((1 - ALPHA) * dice)
        
        return combo

class WeightedCrossEntropyLoss(nn.CrosEntropyLoss):
    def __init__(self, weight=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight
    
    def forward(self, pred, target):
        target = target.long()
        num_classes = pred.size()[1]

        i0 = 1
        i1 = 2

        while i1 < len(pred.shape):
            pred = pred.transpose(i0, i1)
            i0 += 1
            i1 += 1

        pred = pred.contiguous()
        pred = pred.view(-1, num_classes)

        target = target.view(-1,)
        wce_loss = nn.CrosEntropyLoss(weight=self.weight)

        return wce_loss(pred, target)
