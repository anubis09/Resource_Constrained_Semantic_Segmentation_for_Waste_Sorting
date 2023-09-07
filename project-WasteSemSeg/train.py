import os
import random

import torch
from torch import optim
from torch.autograd import Variable
from torch.nn import NLLLoss2d
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import save_image
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

from losses import set_loss

from config import cfg
from loading_data import loading_data
from utils import *
from timer import Timer
import pdb
from tqdm import tqdm

exp_name = cfg.TRAIN.EXP_NAME
log_txt = cfg.TRAIN.EXP_LOG_PATH + '/' + exp_name + '.txt'
writer = SummaryWriter(cfg.TRAIN.EXP_PATH+ '/' + exp_name)

pil_to_tensor = standard_transforms.ToTensor()
train_loader, val_loader, restore_transform = loading_data()


def main(net_name = 'Enet', loss_name = 'Cross_Entropy', checkpoint = False):

    net_name = net_name.lower()

    save_every = 10
    start_epoch = 0

    cfg_file = open('./config.py',"r")  
    cfg_lines = cfg_file.readlines()
    
    with open(log_txt, 'a') as f:
            f.write(''.join(cfg_lines) + '\n\n\n\n')
    if len(cfg.TRAIN.GPU_ID)==1:
        torch.cuda.set_device(cfg.TRAIN.GPU_ID[0])
    torch.backends.cudnn.benchmark = True
    
    net = set_net(net_name)  
    print(f"Net successufully set to: {net_name}")  

    if len(cfg.TRAIN.GPU_ID)>1:
        net = torch.nn.DataParallel(net, device_ids=cfg.TRAIN.GPU_ID).cuda()
    else:
        net=net.cuda()

    net.train()
    #criterion = torch.nn.BCEWithLogitsLoss().cuda() # Binary Classification
    criterion = set_loss(loss_name)
    print(f"criterion successufully set to: {loss_name if loss_name != "" else "Cross-entropy"}")
    optimizer = optim.Adam(net.parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=cfg.TRAIN.NUM_EPOCH_LR_DECAY, gamma=cfg.TRAIN.LR_DECAY)
    _t = {'train time' : Timer(),'val time' : Timer()} 

    # Validation lists for plots
    mIoU_list = []
    aluminium_mIoU_list = []
    paper_mIoU_list = []
    bottle_mIoU_list = []
    nylon_mIoU_list = []

    if checkpoint:
        # load the checkpoint net.
        net, optimizer, scheduler, start_epoch, mIoU_list, aluminium_mIoU_list, paper_mIoU_list, bottle_mIoU_list, nylon_mIoU_list = load_checkpoints(net_name, net, optimizer, scheduler)

    print() 
    print(f'Initial mIoU NO TRAINING: ', end='')

    validate(val_loader, net, criterion, optimizer, -1, restore_transform)

    print('\n')
   
    for epoch in range(start_epoch, start_epoch+cfg.TRAIN.MAX_EPOCH):

        _t['train time'].tic()
        train(train_loader, net, criterion, optimizer, scheduler, epoch)
        _t['train time'].toc(average=False)
        print('🟠 TRAINING time of epoch {}/{} = {:.2f}s'.format(epoch+1, start_epoch+cfg.TRAIN.MAX_EPOCH, _t['train time'].diff))
        print("learning rate: ",optimizer.param_groups[0]['lr'])
        _t['val time'].tic()
        mIoU, aluminium_mIoU, paper_mIoU, bottle_mIoU, nylon_mIoU = validate(val_loader, net, criterion, optimizer, epoch, restore_transform)
        mIoU_list.append(mIoU)
        aluminium_mIoU_list.append(aluminium_mIoU)
        paper_mIoU_list.append(paper_mIoU)
        bottle_mIoU_list.append(bottle_mIoU)
        nylon_mIoU_list.append(nylon_mIoU)
        _t['val time'].toc(average=False)
        print('🟢 VALIDATION time of epoch {}/{} = {:.2f}s'.format(epoch+1, start_epoch+cfg.TRAIN.MAX_EPOCH,  _t['val time'].diff))
            
        # save the model state every 10 epochs 
        if (epoch+1) % save_every == 0:
            checkpoint = {
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                #'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch+1,
                'mIoU_list': mIoU_list,
                'aluminium_mIoU_list': aluminium_mIoU_list,
                'paper_mIoU_list': paper_mIoU_list,
                'bottle_mIoU_list': bottle_mIoU_list,
                'nylon_mIoU_list': nylon_mIoU_list
            }
            torch.save(checkpoint, f'checkpoints/{net_name}/checkpoint_{net_name}_N_CLASSES={cfg.DATA.NUM_CLASSES}_epoch={epoch+1}.pth')
            print(f"🔷 Model checkpoint '{f'checkpoint_{net_name}_N_CLASSES={cfg.DATA.NUM_CLASSES}_epoch={epoch+1}.pth'}' saved")
            if epoch >= start_epoch+save_every:
                os.remove(f'checkpoints/{net_name}/checkpoint_{net_name}_N_CLASSES={cfg.DATA.NUM_CLASSES}_epoch={epoch+1-save_every}.pth')

    return mIoU_list, aluminium_mIoU_list, paper_mIoU_list, bottle_mIoU_list, nylon_mIoU_list


def train(train_loader, net, criterion, optimizer, scheduler, epoch):

    train_progress = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1} Training", leave=False)

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()
   
        optimizer.zero_grad()
        outputs = net(inputs)
        #loss = criterion(outputs, labels.unsqueeze(1).float()) # Binary Segmentation
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_progress.update(1)
    
    train_progress.close()


def validate(val_loader, net, criterion, optimizer, epoch, restore):
    net.eval()
    criterion.cpu()
    input_batches = []
    output_batches = []
    label_batches = []
    iou_ = 0.0
    iou_classes_=[0,0,0,0,0]
    validation_progress = tqdm(total=len(val_loader), desc=f"Epoch {epoch+1} Validation", leave=False)
    for vi, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs, volatile=True).cuda()
        labels = Variable(labels, volatile=True).cuda()
        outputs = net(inputs)
        #for binary classification
        # outputs[outputs>0.5] = 1
        # outputs[outputs<=0.5] = 0

        softmax = nn.Softmax(dim=1)
        outputs = torch.argmax(softmax(outputs),dim=1)
  
        iou, iou_classes = calculate_mean_iu([outputs.squeeze_(1).data.cpu().numpy()], [labels.data.cpu().numpy()], cfg.DATA.NUM_CLASSES)
        iou_ += iou
        iou_classes_ = [sum(x) for x in zip(iou_classes_, iou_classes)]

        validation_progress.update(1)
    
    validation_progress.close()
    mean_iu = iou_/len(val_loader)
    iou_classes_ = [x / len(val_loader) for x in iou_classes_]

    print('[avg mean IoU =  %.4f]' % (mean_iu))
    print(f'mIoU C1 (Aluminium) = {round(iou_classes_[0], 4)}   mIoU C2 (Paper) = {round(iou_classes_[1], 4)}   mIoU C3 (Bottle) = {round(iou_classes_[2], 4)}   mIoU C4 (Nylon) = {round(iou_classes_[3], 4)}')

    net.train()
    criterion.cuda()

    return mean_iu, iou_classes_[0], iou_classes_[1], iou_classes_[2], iou_classes_[3]


if __name__ == '__main__':
    main()
