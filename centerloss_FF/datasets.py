import csv
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pickle
import os

class ImageDataset(Dataset):
    def __init__(self, csv_file, img_size):
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size,img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        
        self.img = []
        self.label = []

        check_file = os.path.join('FF_check_face', 'c23_all.pkl')
        try:
            with open(check_file, 'rb') as f:
                check_face = pickle.load(f)
        except:
            check_face = 'False'

        with open(csv_file, newline='') as csvfile:
            rows= csv.reader(csvfile, delimiter=',')
            for row in rows:
                img_path = row[0]
                mylabel = int(row[1])
                video = img_path.split('/')[-3] + '_' + img_path.split('/')[-2]
                frame = (img_path.split('/')[-1]).split('.')[0]
                try:
                    if mylabel == 0 or check_face[video][frame]:
                        self.img.append(img_path)
                        self.label.append(mylabel)
                except:
                    self.img.append(img_path)
                    self.label.append(mylabel)
                '''
                self.img.append(img_path)
                self.label.append(mylabel)
                ''' 
    def __getitem__(self, index):
        
        path = self.img[index % len(self.img)]

        img = Image.open(path)
        label = self.label[index % len(self.label)]
        video_name = path.split('/')[-3] + '-' + path.split('/')[-2]
        frame = path.split('/')[-1]

        img= self.transform(img)

        return {"img": img, "label": label, "video_name": video_name, "frame": frame, "path": path}

    def __len__(self):
        return len(self.img)

    def combine(self, to_combine):
        
        self.img = (self.img + to_combine.img)
        self.label = (self.label + to_combine.label)
        

class ImageDataset_Test(Dataset):
    def __init__(self, csv_file, img_size, filter_size, test_set):
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size,img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.img = []
        self.label = []
        
        comp = test_set.split('_')[0]
        check_file = os.path.join('FF_check_face_test_%s'%filter_size, comp, '%s.pkl'%test_set)
        try:
            with open(check_file, 'rb') as f:
                check_face = pickle.load(f)
        except:
            check_face = 'False'

        with open(csv_file, newline='') as csvfile:
            rows = csv.reader(csvfile, delimiter=',')
            for row in rows:
                img_path = row[0]
                mylabel = int(row[1])
                video = img_path.split('/')[-4] + '_' + img_path.split('/')[-3]
                num  = (img_path.split('/')[-2])
                frame = (img_path.split('/')[-1]).split('.')[0]
                
                try:
                    if check_face[video][frame][num]:
                        self.img.append(img_path)
                        self.label.append(mylabel)
                except:
                    self.img.append(img_path)
                    self.label.append(mylabel)
                '''
                self.img.append(img_path)
                self.label.append(mylabel)
                '''
    def __getitem__(self, index):

        path = self.img[index % len(self.img)]

        img = Image.open(path)
        label = self.label[index % len(self.label)]
        video_name = path.split('/')[-4] + '-' + path.split('/')[-3]
        frame = path.split('/')[-1]

        img= self.transform(img)

        return {"img": img, "label": label, "video_name": video_name, "frame": frame, "path": path}

    def __len__(self):
        return len(self.img)
