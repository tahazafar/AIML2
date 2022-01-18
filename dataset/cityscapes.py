import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import json


class CityscapesDataset(data.Dataset):
    def __init__(self, root, max_iters=None, train=True, crop_size=(1024, 512), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.train = train
        self.set = 'train' if self.train else 'val'
        self.is_mirror = mirror
        self.files = []

        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        #self.img_ids = [i_id.strip() for i_id in open(list_path)]
        #if not max_iters==None:
        #    self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        #self.files = []

        if self.train:
            self.img_ids = [i_id.strip() for i_id in open(os.path.join(self.root, "train.txt"))]
        else:
            self.img_ids = [i_id.strip() for i_id in open(os.path.join(self.root, "val.txt"))]
        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.info = json.load(open(os.path.join(root, "info.json"), 'r'))
        self.class_mapping = self.info['label2train']

        self.id_to_trainid = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12,
                              26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}


        
        for name in self.img_ids:
          img_file = self.root + "/images/" + name.split("/")[1]
          label_file = self.root + "/labels/" + name.split("/")[1].replace("leftImg8bit", "gtFine_labelIds")
          self.files.append({
              "img": img_file,
              "label": label_file,
              "name": name
          })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        
        #open image and label file
        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)
        
        #convert into numpy array
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # re-assign labels to match the format of Cityscapes
        label_copy = 255 * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))
        label = label_copy

        return image.copy(), label
