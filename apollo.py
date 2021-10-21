#!/usr/bin/python
# -*- encoding: utf-8 -*-


from genericpath import isfile
import torch
from torch.utils import data
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import os.path as osp
import os
from PIL import Image
import numpy as np
import json
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from transform import *

matchLabels = [
    # name id trainId category catId hasInstances ignoreInEval color
    ( 'void' , 0 , 0, 'void' , 0 , False , False , ( 0, 0, 0) ),
    ( 's_w_d' , 200 , 1 , 'dividing' , 1 , False , False , ( 70, 130, 180) ),
    ( 's_y_d' , 204 , 1 , 'dividing' , 1 , False , False , (220, 20, 60) ),
    ( 'ds_w_dn' , 213 , 1 , 'dividing' , 1 , False , True , (128, 0, 128) ),
    ( 'ds_y_dn' , 209 , 1 , 'dividing' , 1 , False , False , (255, 0, 0) ),
    ( 'sb_w_do' , 206 , 1 , 'dividing' , 1 , False , True , ( 0, 0, 60) ),
    ( 'sb_y_do' , 207 , 1 , 'dividing' , 1 , False , True , ( 0, 60, 100) ),
    ( 'b_w_g' , 201 , 2 , 'guiding' , 2 , False , False , ( 0, 0, 142) ),
    ( 'b_y_g' , 203 , 2 , 'guiding' , 2 , False , False , (119, 11, 32) ),
    ( 'db_w_g' , 211 , 2 , 'guiding' , 2 , False , True , (244, 35, 232) ),
    ( 'db_y_g' , 208 , 2 , 'guiding' , 2 , False , True , ( 0, 0, 160) ),
    ( 'db_w_s' , 216 , 3 , 'stopping' , 3 , False , True , (153, 153, 153) ),
    ( 's_w_s' , 217 , 3 , 'stopping' , 3 , False , False , (220, 220, 0) ),
    ( 'ds_w_s' , 215 , 3 , 'stopping' , 3 , False , True , (250, 170, 30) ),
    ( 's_w_c' , 218 , 4 , 'chevron' , 4 , False , True , (102, 102, 156) ),
    ( 's_y_c' , 219 , 4 , 'chevron' , 4 , False , True , (128, 0, 0) ),
    ( 's_w_p' , 210 , 5 , 'parking' , 5 , False , False , (128, 64, 128) ),
    ( 's_n_p' , 232 , 5 , 'parking' , 5 , False , True , (238, 232, 170) ),
    ( 'c_wy_z' , 214 , 6 , 'zebra' , 6 , False , False , (190, 153, 153) ),
    ( 'a_w_u' , 202 , 7 , 'thru/turn' , 7 , False , True , ( 0, 0, 230) ),
    ( 'a_w_t' , 220 , 7 , 'thru/turn' , 7 , False , False , (128, 128, 0) ),
    ( 'a_w_tl' , 221 , 7 , 'thru/turn' , 7 , False , False , (128, 78, 160) ),
    ( 'a_w_tr' , 222 , 7 , 'thru/turn' , 7 , False , False , (150, 100, 100) ),
    ( 'a_w_tlr' , 231 , 7 , 'thru/turn' , 7 , False , True , (255, 165, 0) ),
    ( 'a_w_l' , 224 , 7 , 'thru/turn' , 7 , False , False , (180, 165, 180) ),
    ( 'a_w_r' , 225 , 7 , 'thru/turn' , 7 , False , False , (107, 142, 35) ),
    ( 'a_w_lr' , 226 , 7 , 'thru/turn' , 7 , False , False , (201, 255, 229) ),
    ( 'a_n_lu' , 230 , 7 , 'thru/turn' , 7 , False , True , (0, 191, 255) ),
    ( 'a_w_tu' , 228 , 7 , 'thru/turn' , 7 , False , True , ( 51, 255, 51) ),
    ( 'a_w_m' , 229 , 7 , 'thru/turn' , 7 , False , True , (250, 128, 114) ),
    ( 'a_y_t' , 233 , 7 , 'thru/turn' , 7 , False , True , (127, 255, 0) ),
    ( 'b_n_sr' , 205 , 8 , 'reduction' , 8 , False , False , (255, 128, 0) ),
    ( 'd_wy_za' , 212 , 8 , 'attention' , 8 , False , True , ( 0, 255, 255) ),
    ( 'r_wy_np' , 227 , 8 , 'no parking' , 8 , False , False , (178, 132, 190) ),
    ( 'vom_wy_n' , 223 , 8 , 'others' , 8 , False , True , (128, 128, 64) ),
    ( 'om_n_n' , 250 , 8 , 'others' , 8 , False , False , (102, 0, 204) ),
    ( 'noise' , 249 , 0 , 'ignored' , 0 , False , True , ( 0, 153, 153) ),
    ( 'ignored' , 255 , 0 , 'ignored' , 0 , False , True , (255, 255, 255) ),
    ]

def getPath(root, imgs, lbs):
    contents = os.listdir(root)
    for content in contents:
        contPath = os.path.join(root, content)
        if(os.path.isfile(contPath)):
            imgs.append(contPath)
            labels.append(contPath.replace('ColorImages', 'Gray_Label').replace('ColorImage', 'Label').replace('.jpg', '_bin.png'))
        else:
            getPath(contPath, imgs, labels)

class Apollo(Dataset):
    def __init__(self, rootpth, cropsize=(2048, 1024), mode='train', 
    randomscale=(0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0), *args, **kwargs):
        super(Apollo, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test', 'trainval')
        self.mode = mode
        print('self.mode', self.mode)
        self.ignore_lb = 255

        self.data = pd.read_csv(rootpth + '/' + self.mode + '.csv', header=None, names=["image","label"])
        self.imgs = self.data["image"].values[1:]
        self.labels = self.data["label"].values[1:]
        self.ids = []
        self.trainIds = []

        self.len = len(self.imgs)

        for info in matchLabels:
            self.ids.append(info[1])
            self.trainIds.append(info[2])
        self.lb_map = dict(zip(self.ids, self.trainIds))
        

        # pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.trans_train = Compose([
            ColorJitter(
                brightness = 0.5,
                contrast = 0.5,
                saturation = 0.5),
            HorizontalFlip(),
            RandomScale(randomscale),
            RandomCrop(cropsize)
            ])


    def __getitem__(self, idx):
        impth = self.imgs[idx]
        lbpth = self.labels[idx]
        img = Image.open(impth).convert('RGB')
        label = Image.open(lbpth)
        if self.mode == 'train' or self.mode == 'trainval':
            im_lb = dict(im = img, lb = label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        label = self.convert_labels(label)
        return img, label


    def __len__(self):
        return self.len


    def convert_labels(self, label):
        for k, v in self.lb_map.items():
            label[label == k] = v
        return label



if __name__ == "__main__":
    # path of baidu's colorimgs data
    datapth = "/home/cyr/perception/Baidu/ColorImages"
    if not osp.exists('apolloData'):
        os.mkdir('apolloData')
        images, labels = [], []
        getPath(root=datapth, imgs=images, lbs=labels)
        total = len(images)
        six_part = int(total * 0.6)
        eight_part = int(total * 0.8)
        all=pd.DataFrame({'image': images, 'label': labels})
        allshuffled = shuffle(all)
        allshuffled[:six_part].to_csv('apolloData/train.csv', index=False)
        allshuffled[six_part:eight_part].to_csv('apolloData/val.csv', index=False)
        allshuffled[eight_part:].to_csv('apolloData/test.csv', index=False)

    #load the data and visualize the label with a palette
    ds = Apollo('apolloData', mode="val")  
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    for img, lb in ds:
        r = Image.fromarray(np.uint8(torch.from_numpy(lb).squeeze(0).numpy()))
        r.putpalette(colors)
        plt.cla
        plt.imshow(r)
        plt.pause(0.1)
        

        
    

