"""
Author: Andreas Rössler
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import xception 
import math
import datetime

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import datasets
import torchvision
from datasets import *

import csv
import time
from sklearn.metrics import log_loss, roc_auc_score
import multiprocessing
import pickle
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class COCOLoss(nn.Module):
    """
        Refer to paper:
        Yu Liu, Hongyang Li, Xiaogang Wang
        Rethinking Feature Discrimination and Polymerization for Large scale recognition. NIPS workshop 2017
        re-implement by yirong mao
        2018 07/02
        """

    def __init__(self, num_classes, feat_dim, alpha=6.25):
        super(COCOLoss, self).__init__()

        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.alpha = alpha
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feat):
        norms = torch.norm(feat, p=2, dim=-1, keepdim=True)
        nfeat = torch.div(feat, norms)
        snfeat = self.alpha*nfeat

        norms_c = torch.norm(self.centers, p=2, dim=-1, keepdim=True)
        ncenters = torch.div(self.centers, norms_c)

        logits = torch.matmul(snfeat, torch.transpose(ncenters, 0, 1))

        return logits

def xcept_pretrained_model(numClasses):
    model = xception.xception()
    numFtrs = model.last_linear.in_features
    model.last_linear = COCOLoss(numClasses, numFtrs)
    return model

def save_feature(feature, img_path, feature_root_folder):
    
    ftr = feature.cpu().numpy()
    feature_folder = os.path.join(feature_root_folder, '/'.join(img_path.split('/')[-4:-1]))

    if not os.path.exists(feature_folder):
        os.makedirs(feature_folder)
    #os.system('mkdir -p {}'.format(feature_folder))

    pkl_path = os.path.join(feature_root_folder, '/'.join(img_path.split('/')[-4:]).replace('jpg', 'pkl'))
    with open(pkl_path, 'wb') as f:
        pickle.dump(ftr, f)

def check_frame(prob_list, label, fake_threshold):
    result = 0
    fake_list = []
    max_real = 0
    max_vector = np.zeros(opt.num_out_classes)
    for prob in prob_list:
        if np.argmax(prob, axis=0) != 0 and max(prob) > 0.7:
            fake_list.append(prob)
        elif max(prob) > max_real:
            max_real = max(prob)
            max_vector = prob

    if not fake_list:
        result = 0
    else:
        maxima = 0
        for i in fake_list:
            if max(i) > maxima:
                maxima = max(i)
                max_vector = i
                result = np.argmax(i)
    return max_vector, result==label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='xception')
    parser.add_argument("--train_set", type=str, help="e.g.c23_NT")
    parser.add_argument("--test_set", type=str, help="DF, FS, F2F, NT or all")
    parser.add_argument("--input_size", type=int, default=299)
    parser.add_argument("--batch_size", type=int, default=30, help="size of the batches")
    parser.add_argument("--num_out_classes", type=int)
    parser.add_argument("--filter_size", type=str)
    parser.add_argument("--model_summary", type=bool, default=False)
    parser.add_argument("--loss_type", type=str)
    parser.add_argument("--featureSaved", dest='featureSaved', action='store_true', default=False)
    parser.add_argument("--feature_root_folder", type=str)
    parser.add_argument("--info_folder", type=str)
    parser.add_argument("--fake_threshold", type=float, default=0.3)
    parser.add_argument("--probabilitySaved", dest='probabilitySaved', action='store_true', default=False)
    parser.add_argument("--dataset_folder", type=str)

    opt = parser.parse_args()
    featureSaved = opt.featureSaved
    
    cuda = True if torch.cuda.is_available() else False

    model = xcept_pretrained_model(opt.num_out_classes)

    if cuda:
        model.cuda()
    print("pretrained:", model.state_dict()["last_linear.centers"])

    '''    
    coco_dict = torch.load("/home/aiiulab/coco/train_coco_model/finetunec401.pth")

    
    ft_dict = {}
    for k, v in coco_dict.items():
        if k.split('.')[0] == 'module' or k.split('.')[0] == 'model':
            ft_dict['.'.join(k.split('.')[1:])] = v
        elif k.split('.')[-1] == 'centers':
            ft_dict['.'.join(k.split('.')[0:-1])] = v
        else:
            ft_dict[k] = v
    
    ft_dict = {k: v for k, v in coco_dict.items() if k == "last_linear.centers"}
    ft_dict = ft_dict["last_linear.centers"].to(device)
    print("fintuned:", ft_dict)
    '''
    
    model.load_state_dict(torch.load("/home/aiiulab/coco/final_coco_model/COCOLoss_c23_v2.pth"))

    print("loaded:", model.state_dict()["last_linear.centers"])
    '''
    model_dict = model.state_dict()

    model_dict["last_linear.centers"] = ft_dict
    print("replaced:",  model_dict["last_linear.centers"])

    model.load_state_dict(model_dict)
    #print('loading from /home/aiiulab/coco/train_coco_model/model40ALL2.pth')
    '''

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


    test_dataset = ImageDataset_Test("../dataset/%s/test.csv" %opt.test_set , opt.input_size, opt.filter_size, opt.test_set)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)


    ############################
    ##Testing
    ############################
    #fake = 1
    #real = 0

    print('Testing stage')
    print('-' * 10)
    print('%d batches int total' %len(test_dataloader))

    if opt.probabilitySaved:
        all_img_prob = {}
        for attack in os.listdir(opt.dataset_folder):
            all_img_prob[attack] = {}
            attack_path = os.path.join(opt.dataset_folder, attack)
            for v in os.listdir(attack_path):
                all_img_prob[attack][v] = {}
                v_path = os.path.join(attack_path, v)
                for track in os.listdir(v_path):
                    all_img_prob[attack][v][track] = {}
                    t_path = os.path.join(v_path, track)
                    for im in os.listdir(t_path):
                        all_img_prob[attack][v][track][im] = np.zeros(opt.num_out_classes)

    
    
    corrects = 0.0
    predict = {}
    start_time = time.time()

    for i, batch in enumerate(test_dataloader):
        bSTime = time.time()
        model.eval()
        #set model input
        img = Variable(batch["img"].type(Tensor))
        label = batch["label"]
        video_name = batch['video_name']
        frame = batch['frame']
        img_path = batch['path']


        for j in range(len(video_name)):
            if video_name[j] not in predict:
                predict[video_name[j]] = {}
                predict[video_name[j]]['label'] = label[j]
                predict[video_name[j]]['total'] = 0
                predict[video_name[j]]['correct'] = 0
                predict[video_name[j]]['out_prob'] = {}
                predict[video_name[j]]['frame'] = {}
                predict[video_name[j]]['score'] = np.zeros(opt.num_out_classes)
            if frame[j] not in predict[video_name[j]]['frame']:
                predict[video_name[j]]['frame'][frame[j]] = []

        with torch.no_grad():
            output, features= model(img)
            print('output shape:{}, feature shape:{}'.format(output.shape, features.shape))
            output = F.softmax(output, dim=1)
            simp_label = label
            for k in range(len(output)):
                predict[video_name[k]]['frame'][frame[k]].append(output[k].cpu().numpy())
                if opt.probabilitySaved:
                
                    all_img_prob[img_path[k].split('/')[-4]][img_path[k].split('/')[-3]][img_path[k].split('/')[-2]][img_path[k].split('/')[-1]] = output[k].cpu().numpy()

                if opt.featureSaved:
                    
                    save_feature(features[k], img_path[k], opt.feature_root_folder)

        bETime = time.time()
        print('#{} batch finished, eclipse time: {}'.format(i, bETime-bSTime))
    
    prob_x = []
    label_y = []

    for video in predict:
        predict[video]['total'] = len(predict[video]['frame'])
        for frame in predict[video]['frame']:
            frm_score_vector, isCorrect = check_frame(predict[video]['frame'][frame], predict[video]['label'], opt.fake_threshold)
            if isCorrect:    #check ok
                predict[video]['correct'] += 1
            prob_x.append(frm_score_vector)
            label_y.append(predict[video]['label'].item())


    prob_x_mat = np.vstack(prob_x)
    label_y_mat = np.zeros((len(label_y), 5))
    print(prob_x_mat.shape)
    print(label_y_mat.shape)

    for i, l in enumerate(label_y):
        label_y_mat[i][l] = 1

    logloss_auc_acc = {}
    logloss_auc_acc['logloss'] = log_loss(label_y, prob_x)
    logloss_auc_acc['auc'] = roc_auc_score(label_y_mat, prob_x_mat)
    print(logloss_auc_acc['auc'])

    '''
    if len(predict[video]['frame'][frame]) == 2:
        max_prob = predict[video]['frame'][frame][0]
        if predict[video]['frame'][frame][1][1] > max_prob[1]:
            max_prob = predict[video]['frame'][frame][1]
        predict[video]['out_prob'][frame] = max_prob
    else:
        predict[video]['out_prob'][frame] = predict[video]['frame'][frame][0]
    '''
    '''
    #dump output of testing data
    output_dir = 'out_prob/%s/%s_%s/%s/%s_epoch' %(opt.model, opt.train_set, opt.test_set, opt.loss_type, opt.epoch)
    os.makedirs(output_dir, exist_ok=True)
    for video_name in predict:
        with open(os.path.join(output_dir, video_name + '.csv'), 'wb') as out_file:
            writer = csv.writer(out_file, delimiter='   ')
            for x in predict[video_name]['out_prob']:
                writer.writerow(x, 1-predict[video_name]['label'], predict[video_name]['label'], \
                        predict[video_name]['out_prob'][x][0], predict[video_name]['out_prob'][x][1]) #frame  label  probability
    '''
    #dump video accuracy
    video_acc_dir = 'video_acc'
    os.makedirs(video_acc_dir, exist_ok=True)
    with open(os.path.join(video_acc_dir, "video_acc_%s_%s.csv" %(opt.train_set, opt.test_set)), 'w') as output_file:
        writer = csv.writer(output_file, delimiter=',')
        writer.writerow(["video_name", "correct_frame", "total_frame"])
        for video_name in predict:
            writer.writerow([video_name, predict[video_name]['correct'], len(predict[video_name]['frame'])])


    processed_predict = {}
    if opt.test_set.split('_')[1] == 'all':
        test_list = ['original', 'FS', 'F2F', 'DF', 'NT']
    else:
        test_list = ['original']
        test_list.append(opt.test_set)

    for test_type in test_list:
        processed_predict[test_type] = {}
        processed_predict[test_type]['total'] = 0
        processed_predict[test_type]['correct'] = 0

    for n in predict:
        if predict[n]['correct'] > predict[n]['total'] - predict[n]['correct']:
            corrects += 1

            if n.find("original") != -1:
                processed_predict['original']['correct'] += 1
            elif n.find("NT") != -1:
                processed_predict['NT']['correct'] += 1
            elif n.find("F2F") != -1:
                processed_predict['F2F']['correct'] += 1
            elif n.find("FS") != -1:
                processed_predict['FS']['correct'] += 1
            elif n.find("DF") != -1:
                processed_predict['DF']['correct'] += 1

        if n.find("original") != -1:
            processed_predict['original']['total'] += 1
        elif n.find("NT") != -1:
            processed_predict['NT']['total'] += 1
        elif n.find("F2F") != -1:
            processed_predict['F2F']['total'] += 1
        elif n.find("FS") != -1:
            processed_predict['FS']['total'] += 1
        elif n.find("DF") != -1:
            processed_predict['DF']['total'] += 1
    end = time.time()
        
        
    test_acc = corrects/float(len(predict))
    logloss_auc_acc['acc'] = test_acc

    # save label & probability matrices, logloss, AUC, probability dict
    os.system('mkdir -p {}'.format(opt.info_folder))
    with open(os.path.join(opt.info_folder, 'test_face_prob.pkl'), 'wb') as f:
        pickle.dump(prob_x, f)
    with open(os.path.join(opt.info_folder, 'test_frame_gt.pkl'), 'wb') as f:
        pickle.dump(label_y, f)
    with open(os.path.join(opt.info_folder, 'logloss_auc.pkl'), 'wb') as f:
        pickle.dump(logloss_auc_acc, f)
    if opt.probabilitySaved:
        with open(os.path.join(opt.info_folder, 'img_prob_dict.pkl'), 'wb') as f:
            pickle.dump(all_img_prob, f)

    print()
    print('running time: {}'.format(end-start_time))
    print('train on {}, test on {}'.format(opt.train_set, opt.test_set))
    print('Correct prediction: {}, video_num: {}, Acc: {:.4f}'.format(corrects, len(predict), test_acc))
    print()

    for test_type in test_list:
        correct_video = processed_predict[test_type]['correct']
        total_video = processed_predict[test_type]['total']
        acc = correct_video/float(total_video)
        print('{} correct prediction: {}, video_num: {}, Acc: {:.4f}'.format(test_type, correct_video, total_video, acc))
        
    print()
    print('-' * 10)
    


