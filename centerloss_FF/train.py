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
import os.path as osp
from torch.utils.data import Dataset, DataLoader
from utils import Logger
from xception import xception
from transform import xception_default_data_transforms as data_transforms
from torchvision import transforms, utils, models, datasets
import pandas as pd
from PIL import Image
import time
import torch.backends.cudnn as cudnn

from center_loss import CenterLoss


# center loss optimize
import argparse
parser = argparse.ArgumentParser("Center Loss Example")
parser.add_argument('--weight-cent', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--seed', type=int, default=5)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--use-cpu', action='store_true')
args = parser.parse_args()




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
face_dataset = {x: FaceForensicsDataset('{}.csv'.format(x), data_transforms[x], x) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(face_dataset[x], batch_size=20, shuffle=True) for x in ['train', 'val']}
dataset_sizes = {x: len(face_dataset[x]) for x in ['train', 'val']}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# initial net
'''
def set_parameter_requires_grad(model, featureExtracting):
    if featureExtracting:
        for param in model.parameters():
            param.requires_grad = True
'''

def xcept_pretrained_model(numClasses, featureExtract=True, usePretrained=True):
    model = xception()
    #set_parameter_requires_grad(model, featureExtract)
    numFtrs = model.last_linear.in_features
    model.last_linear = nn.Linear(numFtrs, numClasses)
    input_size = 299
    return model

print(xcept_pretrained_model)
    

    
# train
def train(model, criterion, optimizer, scheduler, num_epochs):

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
                #print('labels', labels.shape) = [20]


                with torch.set_grad_enabled(phase == 'train'):
                    outputs, feats = model(imgs)
                    feats.to(device)
                    outputs.to(device)
                    #print('features:', feats.shape) #= [20, 2048]
                    #print('outputs:', outputs.shape) #= [20, 5]
                    
                    loss_cent = criterion[1](feats, labels)
                    loss_entropy = criterion[0](outputs, labels)
                    loss_cent *= args.weight_cent

                    loss = loss_entropy + loss_cent

                    _, preds = torch.max(outputs, 1)
                    

                    if phase == 'train':
                        optimizer[0].zero_grad()
                        optimizer[1].zero_grad()

                        loss.backward()
                        optimizer[0].step()
                        # by doing so, weight_cent would not impact on the learning of centers
                        for param in criterion[1].parameters():
                            param.grad.data *= (1. / args.weight_cent)
                        optimizer[1].step()
                        


                print('#{} batch with loss: {}'.format(idx, loss.item() * imgs.size(0)), 
                      '#{} batch corrects: {}'.format(idx, torch.sum(preds == labels.data)), 
                      '#{} batch accuracy: {}%'.format(idx, torch.sum(preds == labels.data)*100. /20))
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
        temp_model = f'./train_center_model/tunemodelALL{epoch}.pth'
        torch.save(model.state_dict(), temp_model)
        
    print()
    print('-' * 10)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return model

#ipdb.set_trace()





def main():

    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    sys.stdout = Logger(osp.join('./log_centerALLtune.txt'))

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    
    model_ft = xcept_pretrained_model(numClasses=5, featureExtract=True, usePretrained=True)
    #model_ft.cuda()
    if use_gpu:
        model_ft = nn.DataParallel(model_ft).cuda()


    centerloss = CenterLoss(num_classes=5, feat_dim=2048, use_gpu=True).cuda()
    criterion = [nn.CrossEntropyLoss().cuda(), centerloss]

    # optimize
    params_to_update = model_ft.parameters()
    optimizer4nn = optim.SGD(params_to_update, lr=0.001, momentum=0.9, weight_decay=5e-03)
    optimizer4center = optim.SGD(centerloss.parameters(), lr=0.5, momentum=0.9)
    optimizer = [optimizer4nn, optimizer4center]
    print(params_to_update, optimizer)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer4nn, step_size=5, gamma=0.5)
    
    model_ft = train(model_ft, criterion, optimizer, exp_lr_scheduler, num_epochs=10)
    print(model_ft)
    
    dest_model = './final_center_model/tuneCenterModelALL.pth'
    torch.save(model_ft.state_dict(), dest_model)

#ipdb.set_trace()


if __name__=='__main__':
    main()
#ipdb.set_trace()