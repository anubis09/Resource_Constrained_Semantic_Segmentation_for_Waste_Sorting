import torch
import inspect
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import numpy as np
from PIL import Image
import os
import shutil
from config import cfg

from torch import optim

from model import ENet
from model_custom import ENet as ENet_c
from bisenet import BiSeNetV2
# https://github.com/osmr/imgclsmob/blob/master/pytorch/pytorchcv/models/icnet.py
from icnet import icnet_resnetd50b_cityscapes as icnet

import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 200
plt.rcParams["figure.figsize"] = (10, 6)


def weights_init_kaiming(m):
    if isinstance(m, nn.Conv2d):
        # kaiming is first name of author whose last name is 'He' lol
        nn.init.kaiming_uniform(m.weight)
        m.bias.data.zero_()


def adjust_learning_rate(lr, decay, optimizer, cur_epoch, n_epochs):
    """Sets the learning rate to the initially 
        configured `lr` decayed by `decay` every `n_epochs`"""
    new_lr = lr * (decay ** (cur_epoch // n_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def calculate_mean_iu(predictions, gts, num_classes):
    sum_iu = 0
    iu_classes = []
    for i in range(num_classes):
        n_ii = t_i = sum_n_ji = 1e-9
        for p, gt in zip(predictions, gts):
            n_ii += np.sum(gt[p == i] == i)
            t_i += np.sum(gt == i)
            sum_n_ji += np.sum(p == i)
        iou_i = float(n_ii) / (t_i + sum_n_ji - n_ii)
        sum_iu += iou_i
        iu_classes.append(iou_i)
    mean_iu = sum_iu / num_classes
    return mean_iu, iu_classes


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


def rmrf_mkdir(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.mkdir(dir_name)


def rm_file(path_file):
    if os.path.exists(path_file):
        os.remove(path_file)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(cfg.VIS.PALETTE_LABEL_COLORS)

    return new_mask

# ============================


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.
      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)

    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {'Overall Acc: \t': acc,
            'Mean Acc : \t': acc_cls,
            'FreqW Acc : \t': fwavacc,
            'Mean IoU : \t': mean_iu, }, cls_iu


# This function takes as input the net name and returs the corresponding model.
def set_net(net_name):
    net_name = net_name.lower()
    if (net_name == 'enet'):
        if cfg.TRAIN.STAGE == 'all':
            net = ENet(only_encode=False)
            if cfg.TRAIN.PRETRAINED_ENCODER != '':
                encoder_weight = torch.load(cfg.TRAIN.PRETRAINED_ENCODER)
                del encoder_weight['classifier.bias']
                del encoder_weight['classifier.weight']
                # pdb.set_trace()
                net.encoder.load_state_dict(encoder_weight)
        elif cfg.TRAIN.STAGE == 'encoder':
            net = ENet(only_encode=True)
    elif (net_name == 'enet_c'):
        if cfg.TRAIN.STAGE == 'all':
            net = ENet_c(only_encode=False)
            if cfg.TRAIN.PRETRAINED_ENCODER != '':
                encoder_weight = torch.load(cfg.TRAIN.PRETRAINED_ENCODER)
                del encoder_weight['classifier.bias']
                del encoder_weight['classifier.weight']
                # pdb.set_trace()
                net.encoder.load_state_dict(encoder_weight)
        elif cfg.TRAIN.STAGE == 'encoder':
            net = ENet_c(only_encode=True)
    elif (net_name == 'bisenet'):
        net = BiSeNetV2(n_classes=cfg.DATA.NUM_CLASSES)
    else:
        net = icnet(in_size=(224, 448), num_classes=cfg.DATA.NUM_CLASSES,
                    pretrained=False, aux=False).eval().cuda()
    return net


# PLOTS UTILS
def showTicksLabels(xticks):
    if len(xticks) <= 100 and len(xticks) > 20:
        xticklabels = ['' if (int(i) % 5 != 0 and int(i) > 1)
                       else str(int(i)) for i in xticks]
    elif len(xticks) > 100:
        xticklabels = ['' if (int(i) % 10 != 0 and int(i) > 1)
                       else str(int(i)) for i in xticks]
    else:
        xticklabels = xticks

    return xticklabels

# This function is useful in the colab to generate the plots of the mIoU values.
def plot_mIoU_validation(net_str, mIoU_list, aluminium_mIoU_list, paper_mIoU_list, bottle_mIoU_list,	nylon_mIoU_list, N_epoch, lr, N_classes):

    # FIG 1

    plt.figure(figsize=(10, 5))

    plt.xlabel(f'epoch')
    plt.ylabel(f'mIoU')
    plt.title(f'{net_str} Validation')

    # plt.xticks([x+1 for x in range(N_epoch)])
    plt.plot([x+1 for x in range(N_epoch)], mIoU_list, marker='o')

    ax = plt.gca()

    ax.set_xticks([x+1 for x in range(N_epoch)])
    xticklabels = showTicksLabels([x+1 for x in range(N_epoch)])
    ax.set_xticklabels(xticklabels)

    plt.draw()

    fig_name = f'{net_str}__N_epoch={N_epoch}_LR={lr}_N_classes={N_classes}_->_MAXmIoU={round(max(mIoU_list), 4)}_LASTmIoU={round(mIoU_list[-1], 4)}'
    format = '.png'
    plt.savefig(fig_name+format, dpi=200)

    plt.show()

    # FIG 2

    print()
    print(f'Plot ylim in [0, 1]')
    print()
    plt.figure(figsize=(10, 5))

    plt.xlabel(f'epoch')
    plt.ylabel(f'mIoU')
    plt.title(f'{net_str} Validation')

    # plt.xticks([x+1 for x in range(N_epoch)])
    plt.plot([x+1 for x in range(N_epoch)], mIoU_list, marker='o')
    plt.ylim(0, 1)

    ax = plt.gca()

    ax.set_xticks([x+1 for x in range(N_epoch)])
    xticklabels = showTicksLabels([x+1 for x in range(N_epoch)])
    ax.set_xticklabels(xticklabels)

    plt.draw()

    fig_name = f'{net_str}__N_epoch={N_epoch}_LR={lr}_N_classes={N_classes}_->_MAXmIoU={round(max(mIoU_list), 4)}_LASTmIoU={round(mIoU_list[-1], 4)}_ylim_01'
    format = '.png'
    plt.savefig(fig_name+format, dpi=200)

    plt.show()

    # FIG 3

    print()
    print(f'Plot 4 classes')
    print()

    plt.figure(figsize=(10, 5))

    plt.xlabel(f'epoch')
    plt.ylabel(f'mIoU')
    plt.title(f'{net_str} Validation')

    # plt.xticks([x+1 for x in range(N_epoch)])
    
    plt.plot([x+1 for x in range(N_epoch)],
             aluminium_mIoU_list, color='purple', label='Aluminium', linewidth=1)  # aluminium_mIoU_list
    # plt.plot([x+1 for x in range(N_epoch)], mIoU_list,
    #          marker='o', label='Total mIoU', zorder=100)
    plt.plot([x+1 for x in range(N_epoch)], paper_mIoU_list,
             color='orange', label='Paper', linewidth=1)  # paper_mIoU_list
    plt.plot([x+1 for x in range(N_epoch)],
             bottle_mIoU_list, color='green', label='Bottle', linewidth=1)  # bottle_mIoU_list
    plt.plot([x+1 for x in range(N_epoch)], nylon_mIoU_list,
             color='red', label='Nylon', linewidth=1)  # nylon_mIoU_list
    
    plt.legend(loc='lower right', prop={'size':11}, ncol=4, handletextpad=0.3)
    ax = plt.gca()

    ax.set_xticks([x+1 for x in range(N_epoch)])
    xticklabels = showTicksLabels([x+1 for x in range(N_epoch)])
    ax.set_xticklabels(xticklabels)

    plt.draw()

    fig_name = f'{net_str}__N_epoch={N_epoch}_LR={lr}_N_classes={N_classes}_->_MAXmIoU={round(max(mIoU_list), 4)}_LASTmIoU={round(mIoU_list[-1], 4)}_5classesplot'
    format = '.png'
    plt.savefig(fig_name+format, dpi=200)

    plt.show()

    # FIG 4

    print()
    print(f'Plot 4 classes, ylim in [0, 1]')
    print()

    plt.figure(figsize=(10, 5))

    plt.xlabel(f'epoch')
    plt.ylabel(f'mIoU')
    plt.title(f'{net_str} Validation')

    # plt.xticks([x+1 for x in range(N_epoch)])
    
    plt.plot([x+1 for x in range(N_epoch)],
             aluminium_mIoU_list, color='purple', label='Aluminium', linewidth=1)  # aluminium_mIoU_list
    # plt.plot([x+1 for x in range(N_epoch)], mIoU_list,
    #          marker='o', label='Total mIoU', zorder=100)
    plt.plot([x+1 for x in range(N_epoch)], paper_mIoU_list,
             color='orange', label='Paper', linewidth=1)  # paper_mIoU_list
    plt.plot([x+1 for x in range(N_epoch)],
             bottle_mIoU_list, color='green', label='Bottle', linewidth=1)  # bottle_mIoU_list
    plt.plot([x+1 for x in range(N_epoch)], nylon_mIoU_list,
             color='red', label='Nylon', linewidth=1)  # nylon_mIoU_list
    
    plt.ylim(0, 1)
    
    plt.legend(loc='lower right', prop={'size':11}, ncol=4, handletextpad=0.3)
    ax = plt.gca()

    ax.set_xticks([x+1 for x in range(N_epoch)])
    xticklabels = showTicksLabels([x+1 for x in range(N_epoch)])
    ax.set_xticklabels(xticklabels)

    plt.draw()

    fig_name = f'{net_str}__N_epoch={N_epoch}_LR={lr}_N_classes={N_classes}_->_MAXmIoU={round(max(mIoU_list), 4)}_LASTmIoU={round(mIoU_list[-1], 4)}_5classesplot_ylim_01'
    format = '.png'
    plt.savefig(fig_name+format, dpi=200)

    plt.show()

# This functions is needed to load checkpoints in the colab notebook. 
def load_checkpoints(net_name, net, optimizer, scheduler):
    if len(os.listdir(f'checkpoints/{net_name}')) > 1:
        # load the saved checkpoint
        path_pth_file = [file for file in os.listdir(
            f'checkpoints/{net_name}') if '.pth' in file][0]
        checkpoint = torch.load(f'checkpoints/{net_name}/{path_pth_file}')

        # restore the state of the model and optimizer
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # all the checkpoints don't have a scheduler_state_dict
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # resume training from the saved epoch
        start_epoch = checkpoint['epoch']

        # save previous mIoU list
        mIoU_list = checkpoint['mIoU_list']
        aluminium_mIoU_list = checkpoint['aluminium_mIoU_list']
        paper_mIoU_list = checkpoint['paper_mIoU_list']
        bottle_mIoU_list = checkpoint['bottle_mIoU_list']
        nylon_mIoU_list = checkpoint['nylon_mIoU_list']

        print(f"✅ Model '{path_pth_file}' Loaded\n")
        return net, optimizer, scheduler, start_epoch, mIoU_list, aluminium_mIoU_list, paper_mIoU_list, bottle_mIoU_list, nylon_mIoU_list

# This function create a net from the checkpoint that must be placed inside the corresponding folder "checkpoits/{net_name}".
def create_checkpoint_net(net_name):
    net_name = net_name.lower()
    net = set_net(net_name)
    if len(cfg.TRAIN.GPU_ID) > 1:
        net = torch.nn.DataParallel(net, device_ids=cfg.TRAIN.GPU_ID).cuda()
    else:
        net = net.cuda()
    net.train()
    optimizer = optim.Adam(net.parameters(), lr=cfg.TRAIN.LR,
                           weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = StepLR(
        optimizer, step_size=cfg.TRAIN.NUM_EPOCH_LR_DECAY, gamma=cfg.TRAIN.LR_DECAY)

    net, optimizer, scheduler, start_epoch, mIoU_list = load_checkpoints(
        net_name, net, optimizer, scheduler)
    return net
