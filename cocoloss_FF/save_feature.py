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
from centerloss.utils import Logger
from xception_cent import xception
from dataset_2.transformALL import xception_default_data_transforms as data_transforms
from torchvision import transforms, utils, models, datasets
import pandas as pd
from PIL import Image
import time
import torch.backends.cudnn as cudnn

import pickle

# center loss optimize
import argparse
parser = argparse.ArgumentParser("Center Loss Example")
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
        video = img_path.split('/')[-2]
        frame = img_path.split('/')[-1].split('.')[0]
        manipulate = img_path.split('/')[-3]

        img = Image.open(img_path)

        label = np.array(self.img_path_label.iloc[idx, 1])

        img = self.transform(img)

        return {'image': img, 'label': label, 'video': video, 'frame': frame, 'mani': manipulate}


def xcept_pretrained_model(numClasses, featureExtract=True, usePretrained=True):
    model = xception()
    #set_parameter_requires_grad(model, featureExtract)
    numFtrs = model.last_linear.in_features
    model.last_linear = nn.Linear(numFtrs, numClasses)
    input_size = 299
    return model

###### load data ######
face_dataset = {x: FaceForensicsDataset('{}.csv'.format(x), data_transforms[x], x) for x in ['trainall', 'valall']}
dataloaders = {x: torch.utils.data.DataLoader(face_dataset[x], batch_size=20, shuffle=True) for x in ['trainall', 'valall']}
dataset_sizes = {x: len(face_dataset[x]) for x in ['trainall', 'valall']}
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



# test self
def feature(model):
    
    since = time.time()
    feature = []


    for phase in ['trainall', 'valall']:
        if phase == 'trainall':
            model.train()
        else:
            model.eval()


        

        for idx, inputBatch in enumerate(dataloaders[phase]):
            #inputBatch = inputBatch.to(device)
            imgs, labels, video, frame, mani = inputBatch['image'], inputBatch['label'], inputBatch['video'], inputBatch['frame'], inputBatch['mani']
            imgs = imgs.type(torch.cuda.FloatTensor).to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(False):
                _, feats = model(imgs)
                feature.append(feats.to(device))

                #print("feature", feature.shape)
                print('#{} batch'.format(idx))

            feature = torch.cat(feature, 0)
            feature = feature.cpu().numpy()
            p = 0
            for i in feature:

                if not os.path.isdir('./features_c23_center/{}'.format(mani[p])):
                    os.mkdir('./features_c23_center/{}'.format(mani[p]))
                if not os.path.isdir('./features_c23_center/{}/{}'.format(mani[p], video[p])):
                    os.mkdir('./features_c23_center/{}/{}'.format(mani[p], video[p]))

                with open('./features_c23_center/{}/{}/{}.pkl'.format(mani[p], video[p], frame[p]), 'wb') as f:
                    pickle.dump(i, f)
                    f.close()
                    p += 1
            
            feature = []

        
    print('-' * 10)
    time_elapsed = time.time() - since
    print('Extracting features complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    sys.stdout = Logger(osp.join('./ExtractFeature_c23.txt'))

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    
    trained_model = xcept_pretrained_model(numClasses=5, featureExtract=True, usePretrained=True)
    if use_gpu:
        trained_model = nn.DataParallel(trained_model).cuda()

    trained_model.load_state_dict(torch.load('./final_center_model/tuneCenterModelALL_01.pth')) #opt model's path

    #result
    model_test = feature(trained_model)
    return model_test


if __name__ == "__main__":
    main()
