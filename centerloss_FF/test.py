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
from datasets_tr import *

import csv
import time
from sklearn.metrics import log_loss, roc_auc_score
import multiprocessing
import pickle
import numpy as np

from Tsne_test import tsne

def xcept_pretrained_model(num):
    model = xception.xception()
    numFtrs = model.last_linear.in_features
    model.last_linear = nn.Linear(numFtrs, num)
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
    parser.add_argument("--doTSNE", type=str, default=True)
    parser.add_argument("--featureSaved", dest='featureSaved', action='store_true', default=False)
    parser.add_argument("--feature_root_folder", type=str)
    parser.add_argument("--info_folder", type=str)
    parser.add_argument("--fake_threshold", type=float, default=0.2)
    parser.add_argument("--probabilitySaved", dest='probabilitySaved', action='store_true', default=False)
    parser.add_argument("--dataset_folder", type=str)

    opt = parser.parse_args()
    featureSaved = opt.featureSaved
    
    cuda = True if torch.cuda.is_available() else False

    model = xcept_pretrained_model(opt.num_out_classes)

    if cuda:
        model.cuda()
    
    state_dict = torch.load("/home/frank/center/final_center_model/noglb_center_raw.pth")
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    print('loading from final_center/noglb_center_raw.pth')


    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


    test_dataset = ImageDataset_Test("../dataset/%s/test.csv" %opt.test_set , opt.input_size, opt.filter_size, opt.test_set)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)


    ############################
    ##Testing
    ############################
    #fake = 1
    #real = 0
    print("%s" %opt.test_set)
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
    feat_tsne = []
    label_tsne = []

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
            #print('output shape:{}, feature shape:{}'.format(output[0].shape, output[1].shape))
            output = F.softmax(output, dim=1)
            simp_label = label


            for k in range(len(output)):
                predict[video_name[k]]['frame'][frame[k]].append(output[k].cpu().numpy())
                if opt.probabilitySaved:
                    all_img_prob[img_path[k].split('/')[-4]][img_path[k].split('/')[-3]][img_path[k].split('/')[-2]][img_path[k].split('/')[-1]] = output[k].cpu().numpy()

                if opt.featureSaved:
                    save_feature(features[k], img_path[k], opt.feature_root_folder)

                
                if opt.doTSNE: 
                    feat_tsne.append(features[k])
                    label_tsne.append(simp_label[k])
                    #print(feat_tsne)
                    #print(label_tsne)

        
        bETime = time.time()
        print('#{} batch finished, eclipse time: {}'.format(i, bETime-bSTime))
    
    tsne(feat_tsne, label_tsne)

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
    label_y_mat = np.vstack(label_y)

    logloss_auc_acc = {}
    logloss_auc_acc['logloss'] = log_loss(label_y, prob_x)
    logloss_auc_acc['auc'] = roc_auc_score(label_y_mat, prob_x_mat)

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
    
    #dump video accuracy
    video_acc_dir = 'video_acc'
    os.makedirs(video_acc_dir, exist_ok=True)
    with open(os.path.join(video_acc_dir, "video_acc_%s_%s.csv" %(opt.train_set, opt.test_set)), 'w') as output_file:
        writer = csv.writer(output_file, delimiter=',')
        writer.writerow(["video_name", "correct_frame", "total_frame"])
        for video_name in predict:
            writer.writerow([video_name, predict[video_name]['correct'], len(predict[video_name]['frame'])])
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
