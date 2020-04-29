import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import os 
from torch.utils.data import Dataset, DataLoader
from xception import xception
from transform import xception_default_data_transforms as data_transforms
from torchvision import transforms, utils, models, datasets
import pandas as pd
from PIL import Image
import time
import torch.backends.cudnn as cudnn

import os.path as osp
from utils import Logger
from model_utils import COCOLoss


import argparse
parser = argparse.ArgumentParser("Coco Loss Example")
parser.add_argument('--seed', type=int, default=5)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--use-cpu', action='store_true')
args = parser.parse_args()

csv_pth = ""

# data 
class FaceForensicsDataset(Dataset):
 
    '''
    Data format in .csv file each line:
    /path/to/image.jpg,label
    '''
    def __init__(self, csv_file, transform, state):
        super(FaceForensicsDataset, self).__init__()

        self.img_path_label = pd.read_csv(csv_file)
        self.transform = data_transforms[state]

    def __len__(self):
        return len(self.img_path_label)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = self.img_path_label.iloc[idx, 0]
        img = Image.open(img_path)

        label = np.array(self.img_path_label.iloc[idx, 1])

        img = self.transform(img)

        return {'image': img, 'label': label}

#ipdb.set_trace()

    
###### load data ######
face_dataset = {x: FaceForensicsDataset(csv_pth+'{}.csv'.format(x), data_transforms[x], x) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(face_dataset[x], batch_size=15, shuffle=True) for x in ['train', 'val']}
dataset_sizes = {x: len(face_dataset[x]) for x in ['train', 'val']}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# initial net
def set_parameter_requires_grad(model, featureExtracting):
    if featureExtracting:
        for param in model.parameters():
            param.requires_grad = True

def xcept_pretrained_model(numClasses, featureExtract=True, usePretrained=True):
    model = xception()
    set_parameter_requires_grad(model, featureExtract)
    numFtrs = model.last_linear.in_features
    model.last_linear = COCOLoss(numClasses, numFtrs)
    input_size = 299
    return model
    

    
# train
def train(model, criterion, optimizer, scheduler, num_epochs=8):

    since = time.time()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            total_loss = 0.0
            running_corrects = 0

            for idx, inputBatch in enumerate(dataloaders[phase]):
                #inputBatch = inputBatch.to(device)
                imgs, labels = inputBatch['image'], inputBatch['label']
                imgs = imgs.type(torch.cuda.FloatTensor).to(device)
                labels = labels.to(device)
                #paths = paths.to(device)
                #print(imgs, labels)



                with torch.set_grad_enabled(phase == 'train'):
                    logits, _ = model(imgs)
                    
                    loss = criterion(logits, labels)

                    _, preds = torch.max(logits, 1)
                  
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        

                print('#{} batch with loss: {}'.format(idx, loss.item() * imgs.size(0)), 
                      '#{} batch corrects: {}'.format(idx, torch.sum(preds == labels.data)), 
                      '#{} batch accuracy: {}%'.format(idx, torch.sum(preds == labels.data)*100. /15))
                #print('batch loss: {}'.format(loss.item() * imgs.size(0)))
                total_loss += loss.item() * imgs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                

        if phase == 'train':
            scheduler.step()
            
        epoch_loss = total_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
        
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
        temp_model = f'./train_coco_model/modelc23_{epoch}.pth'
        torch.save(model.state_dict(), temp_model)
        
    print()
    print('-' * 10)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model






def main():

    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False


    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    model_ft = xcept_pretrained_model(numClasses=5, featureExtract=True, usePretrained=True)
    model_ft.cuda()
    
    criterion = nn.CrossEntropyLoss()

    sys.stdout = Logger(osp.join('./log_coco_c23.txt'))

    # optimize
    params_to_update = model_ft.parameters()
    optimizer = optim.SGD(params_to_update, lr=1e-05, momentum=0.9)
    print(params_to_update, optimizer)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    model_ft = train(model_ft, criterion, optimizer, exp_lr_scheduler, num_epochs=8)
    print(model_ft)
    
    dest_model = './final_coco_model/CocoModel_c23.pth'
    torch.save(model_ft.state_dict(), dest_model)

#ipdb.set_trace()


if __name__=='__main__':
    main()
#ipdb.set_trace()
