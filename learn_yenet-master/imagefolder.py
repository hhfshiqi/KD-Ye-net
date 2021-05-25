import torch
import numpy
import os, glob
import random, csv
#import accimage
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

class customData(Dataset):
    def __init__(self, img_path, txt_path, dataset = '',data_transforms=None):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = [os.path.join(img_path, line.strip().split('\t')[0]) for line in lines]
            self.img_label = [int(line.strip().split('\t')[-1]) for line in lines]
        self.data_transforms = data_transforms
        self.dataset = dataset

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(img_name)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms[self.dataset](img)
            except:
                print("Cannot transform image: {}".format(img_name))
        return img, label
#####################以上为图像数据读取，返回（img, label）#########################