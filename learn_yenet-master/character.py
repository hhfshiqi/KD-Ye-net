import torch
import numpy
import os, glob
import random, csv

from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np



class Character(Dataset):

    # root：图片根目录，初始化加载图片信息
    def __init__(self, root,filename,len_images,mode):
        super(Character, self).__init__()

        self.root = root # "name" : 0   name , label
        self.len_images=len_images
        if mode=="train":
            self.filename=filename+"train/"
        # elif mode=="vali":
        #     self.filename = filename + "/valid"
        else:
            self.filename=filename+"valid/"

    def __len__(self):
        return self.len_images

    def __getitem__(self, idx):
        # index∈[0~len(images)]
        # self.images, self.labels
        # img: 'cartoon\\pikachu\\00000023.png'
        # label: int 0,1...
        print(idx)
        if idx>self.len_images/2:
            idx_new =int(idx-self.len_images/2)
            img=self.filename+"stego_s/"+str(idx_new).zfill(5)+".pgm"
            label=[0,1]
            #label = 0
        else:
            img=self.filename+"cover/"+str(idx).zfill(5)+".pgm"
            label=[1,0]
            #label = 1


        #img, label = self.images[idx], self.labels[idx]

        tf = transforms.Compose([
            lambda x: Image.open(x),
            #transforms.Grayscale(),
            transforms.CenterCrop(256),
            transforms.ToTensor()])
        # transforms.Compose([
        #     lambda x: Image.open(x).convert('RGB'),  # string path= > image data
        #     #transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
        #     transforms.Compose([transforms.Grayscale(), transforms.CenterCrop(256), transforms.ToTensor()]
        #     transforms.RandomRotation(15),
        #     transforms.CenterCrop(self.resize),
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        #     #transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #     #                     std=[0.229, 0.224, 0.225])
        # ])

        img = tf(img)  # 把路径转化为图像
        label = torch.tensor(label)

        return img, label
