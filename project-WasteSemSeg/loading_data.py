import torchvision.transforms as standard_transforms
from torch.utils.data import DataLoader
import transforms as own_transforms
from resortit import resortit
from config import cfg

def loading_data():
    mean_std = cfg.DATA.MEAN_STD
    train_simul_transform = own_transforms.Compose([
        own_transforms.Scale(int(cfg.TRAIN.IMG_SIZE[0] / 0.875)),
        own_transforms.RandomCrop(cfg.TRAIN.IMG_SIZE),
        own_transforms.RandomHorizontallyFlip()
    ])
    train_simul_transform_aug = own_transforms.Compose([
        own_transforms.Scale(int(cfg.TRAIN.IMG_SIZE[0] / 0.875)),
        own_transforms.RandomCrop(cfg.TRAIN.IMG_SIZE),
        own_transforms.RandomHorizontallyFlip(),
        own_transforms.RandomVerticalFlip(),
        #own_transforms.RandomColorsJitter()
    ])
    val_simul_transform = own_transforms.Compose([
        own_transforms.Scale(int(cfg.TRAIN.IMG_SIZE[0] / 0.875)),
        own_transforms.CenterCrop(cfg.TRAIN.IMG_SIZE)
    ])
    img_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = standard_transforms.Compose([
        own_transforms.MaskToTensor(),
        own_transforms.ChangeLabel(cfg.DATA.IGNORE_LABEL, cfg.DATA.NUM_CLASSES - 1)
    ])
    restore_transform = standard_transforms.Compose([
        own_transforms.DeNormalize(*mean_std),
        standard_transforms.ToPILImage()
    ])

    # put train_simul_transform_aug to set data augmentation in the training set.
    train_set = resortit('train', simul_transform=train_simul_transform, transform=img_transform,
                           target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=16, shuffle=True)
    
    val_set = resortit('val', simul_transform=val_simul_transform, transform=img_transform,
                         target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=cfg.VAL.BATCH_SIZE, num_workers=16, shuffle=False)

    #return train_loader, train_augmented_loader, val_loader, restore_transform
    return train_loader, val_loader, restore_transform
