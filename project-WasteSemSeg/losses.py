import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from config import cfg
import segmentation_models_pytorch.losses as losses

def set_loss(loss_name):
    loss_name = loss_name.lower()
    match loss_name:
        case "focal":
            loss = losses.FocalLoss("multiclass", gamma = 2).cuda()
            # possible values for gamma : 1, 2, 5 
        case "lovasz":
            loss = losses.LovaszLoss("multiclass").cuda()
        case "dice":
            loss = losses.DiceLoss("multiclass").cuda()
        case "cbfl":
            loss = CB_loss(cfg.DATA.NUM_CLASSES, 0.9, 2).cuda() 
        case "f+l":
            loss = Combined_losses("focal", "lovasz")
        case _:
            loss = torch.nn.CrossEntropyLoss().cuda()
    return loss

class Combined_losses(nn.Module):
    def __init__(self, first_loss, second_loss):
        super().__init__()
        self.first = set_loss(first_loss)
        self.second = set_loss(second_loss)
    
    def forward(self, y_pred, y_true):
        sum = self.first(y_pred, y_true) + self.second(y_pred, y_true)
        return sum

#-------------------

def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.

    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.

    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """    
    BCLoss = F.binary_cross_entropy_with_logits(input = logits, target = labels,reduction = "none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + 
            torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)

    focal_loss /= torch.sum(labels)
    return focal_loss

"""
Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
where Loss is one of the standard losses used for Neural Networks.

Args:
  samples_per_cls: A python list of size [no_of_classes].
  no_of_classes: total number of classes. int
  loss_type: string. One of "sigmoid", "focal", "softmax".
  beta: float. Hyperparameter for Class balanced loss.
  gamma: float. Hyperparameter for Focal loss.

"""
class CB_loss(nn.Module):
    def __init__(self, no_of_classes, beta, gamma):
      super().__init__()
      self.no_of_classes = no_of_classes
      self.beta = beta
      self.gamma = gamma        

    def forward(self, labels, logits):

      samples_per_cls = np.bincount(logits, minlength=self.no_of_classes) #https://github.com/richardaecn/class-balanced-loss/blob/master/data.ipynb
      effective_num = 1.0 - np.power(self.beta, samples_per_cls)
      weights = (1.0 - self.beta) / np.array(effective_num)
      weights = weights / np.sum(weights) * self.no_of_classes

      labels_one_hot = F.one_hot(labels, self.no_of_classes).float()

      weights = torch.tensor(weights).float()
      weights = weights.unsqueeze(0)
      weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
      weights = weights.sum(1)
      weights = weights.unsqueeze(1)
      weights = weights.repeat(1,self.no_of_classes)

      loss_type = "focal"
      if loss_type == "focal":
          cb_loss = focal_loss(labels_one_hot, logits, weights, self.gamma)
      return cb_loss

# Credits https://github.com/vandit15/Class-balanced-loss-pytorch/blob/master/class_balanced_loss.py
