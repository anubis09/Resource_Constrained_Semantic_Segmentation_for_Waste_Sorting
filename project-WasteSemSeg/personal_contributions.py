from utils import create_checkpoint_net
import torch
import torch.nn.utils.prune as prune

import torch
from loading_data import loading_data
from torch.quantization import quantize_fx
import copy
import platform
from torch.autograd import Variable
from tqdm import tqdm
from config import cfg
from utils import calculate_mean_iu
import torch.nn as nn



''' 

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-    PRUNING      #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

'''

# This function takes as input the model, the prune method, the amount of parameters to prune, and two parameters of the ln_unstructured method
# and returns the pruned model. 
def get_pruned_model(model, method=prune.l1_unstructured, amount=0.2, n=2, dim=0):

    N_modules = 0
    N_pruned_modules = 0

    for name, module in model.named_modules():

        # pruning Conv2d -> 72/338
        if isinstance(module, torch.nn.Conv2d):

            if method is prune.random_unstructured or method is prune.l1_unstructured:
                method(module, name='weight', amount=amount)
                N_pruned_modules += 1
            elif method is prune.random_structured:
                method(module, name='weight', amount=amount, dim=dim)
                N_pruned_modules += 1
            elif method is prune.ln_structured:
                method(module, name='weight', amount=amount, n=n, dim=dim)
                N_pruned_modules += 1
            elif method is prune.identity:
                method(module, name='weight')
                N_pruned_modules += 1

        N_modules += 1

    print(f'TOT layers pruned = {N_pruned_modules}/{N_modules}')

    return model


''' 

#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-    QUANTIZATION      #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

'''

# This function takes as input the model, the val loader, and the net name (for the final print) 
# and returns the quantized model. 
def get_quantized_model(model, val_loader, net_str):

    backend = "fbgemm" if "x86" in platform.machine() else "qnnpack"
    m = copy.deepcopy(model)
    m.eval()
    qconfig_dict = {"": torch.quantization.get_default_qconfig(backend)}

    # Prepare
    model_prepared = quantize_fx.prepare_fx(m, qconfig_dict, torch.randn(cfg.VAL.BATCH_SIZE, 3, 224, 448))

    # Calibrate - Use representative (validation) data.
    calibration_progress = tqdm(total=len(val_loader), desc=f"Calibration", leave=False)
    with torch.inference_mode():
        for vi, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs = Variable(inputs, volatile=True).cuda()
            model_prepared(inputs)
            calibration_progress.update(1)

    # quantize
    calibration_progress.close()
    model_prepared = model_prepared.to('cpu')
    model_prepared.eval()
    model_quantized = quantize_fx.convert_fx(model_prepared)

    print(f"✅ {net_str} model quantized from checkpoint") 

    return model_quantized


# This function takes the quantised model and returns the parameters size.
def get_qmodel_param_size(qmodel):

    quantized_dict = qmodel.state_dict()
    total_size = 0
    none_type_counter = 0

    for param_name, param_tensor in quantized_dict.items():
      if('NoneType' not in str(type(param_tensor))):
        param_size = param_tensor.numel() * param_tensor.element_size()
        total_size += param_size
      else:
        none_type_counter += 1

    # Convert total_size to MB for a more human-readable format
    total_size_mb = total_size / (1024**2)

    return (total_size_mb, none_type_counter)

# The validate function is needed because the quantized models only works on cpu. 
# It take as input the val_loader, the net, the criterion and a boolean to know if the model is quantized,
# it returns the mean_IoU of the network over the validate set. 
def validate_(val_loader, net, criterion, is_quantized=False):
    net.eval()
    criterion.cpu()
    iou_ = 0.0
    iou_classes_=[0,0,0,0,0]
    validation_progress = tqdm(total=len(val_loader), desc=f"Validation", leave=False)
    for vi, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs, volatile=True).cuda() if not is_quantized else Variable(inputs, volatile=True)
        labels = Variable(labels, volatile=True).cuda() if not is_quantized else Variable(labels, volatile=True)
        outputs = net.forward(inputs)
        #for binary classification
        # outputs[outputs>0.5] = 1
        # outputs[outputs<=0.5] = 0

        softmax = nn.Softmax(dim=1)
        outputs = torch.argmax(softmax(outputs),dim=1)
  
        iou, iou_classes = calculate_mean_iu([outputs.squeeze_(1).data.cpu().numpy()], [labels.data.cpu().numpy()], cfg.DATA.NUM_CLASSES)
        iou_ += iou
        #iou_classes_ = np.sum(iou_classes, iou_classes_)
        iou_classes_ = [sum(x) for x in zip(iou_classes_, iou_classes)]

        validation_progress.update(1)
    
    validation_progress.close()
    mean_iu = iou_/len(val_loader)
    iou_classes_ = [x / len(val_loader) for x in iou_classes_]

    print()
    print('[avg mean IoU =  %.4f]' % (mean_iu))
    print(f'mIoU C1 (Aluminium) = {round(iou_classes_[0], 4)}   mIoU C2 (Paper) = {round(iou_classes_[1], 4)}   mIoU C3 (Bottle) = {round(iou_classes_[2], 4)}   mIoU C4 (Nylon) = {round(iou_classes_[3], 4)}')

    net.train()
    criterion.cuda()

    return mean_iu
