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
from torchvision import transforms, utils, models, datasets
import pandas as pd
from PIL import Image
import time
import torch.backends.cudnn as cudnn

from tsnecuda import TSNE
import matplotlib.pyplot as plt
import seaborn as sn
import pickle
import numpy as np
import random


# set arguments
import argparse
parser = argparse.ArgumentParser("Center Loss Example")
parser.add_argument('--seed', type=int, default=5)
parser.add_argument('--gpu', type=str, default='yes')
parser.add_argument('--use-cpu', action='store_true')
args = parser.parse_args()

feature_pth = '/home/aiiulab/coco/features/COCOloss_track'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# test self
def tsne():
    
    since = time.time()
    label = []
    feature = []

    for tp in os.listdir(feature_pth):
        tp_pth = os.path.join(feature_pth, tp)
        for vid in os.listdir(tp_pth):
            vid_pth = os.path.join(tp_pth, vid)
            for num in os.listdir(vid_pth):
                num_pth = os.path.join(vid_pth, num)
                for pkls in os.listdir(num_pth):
                    pkl_pth = os.path.join(num_pth, pkls)
                    
                    with open(pkl_pth, 'rb') as f:
                        #frm_featr = pickle.load(f).tolist()
                        frm_featr = pickle.load(f)
                    
                    feature.append(frm_featr)

                    if tp == 'original':
                        label.append(0)
                    elif tp == 'NT':
                        label.append(1)
                    elif tp == 'F2F':
                        label.append(2)
                    elif tp == 'FS':
                        label.append(3)
                    elif tp == 'DF':
                        label.append(4)
            print("{}-{} done".format(tp, vid))
        
    length = len(feature)
    rndm_feat = []
    rndm_label = []

    for i in random.sample(range(0,length),10000):
        rndm_feat.append(feature[i])
        rndm_label.append(label[i])
        
    feature = np.asarray(rndm_feat)
    label = np.asarray(rndm_label)


    model = TSNE(n_components=2, perplexity=30, learning_rate=500, n_iter=1000)
    # configuring the parameteres
    # the number of components = 2 => target reduced dimension
    # default perplexity = 30
    # default learning rate = 200
    # default Maximum number of iterations for the optimization = 1000
    tsne_data = model.fit_transform(feature)

    # creating a new data frame which help us in ploting the result data
    tsne_data = np.vstack((tsne_data.T, label)).T
    tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))
    # Ploting the result of tsne
    g = sn.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
    new_labels = ['real', 'NT', 'F2F', 'FS', 'DF']
    for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
    
    plt.savefig("tsne_coco_raw.png")

    end = time.time()
    print("Making TSNE plot in {}".format(end-since))



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

    tsne()

    


if __name__=='__main__':
    main()


