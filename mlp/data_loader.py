import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageOps
import torch
import torch.utils.data as Data
# import augmentation as aug

import numbers
import os

alphabet = ['<blank>','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i' ,'j' ,'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z','<eos>']

def alpha2Index(alpha_seq):
    Index = []
    for al in alpha_seq:
        Index.append(alphabet.index(al))

    for  i in range(len(Index),5):
        Index.append(alphabet.index('<blank>'))

    Index.append(alphabet.index('<eos>'))

    return torch.LongTensor(Index)


class TrainData(Data.Dataset):

    def __init__(self, file_dir,skip=0):
        super(TrainData, self).__init__()

        self.video_list = []
        self.label_list = []
        self.skipframe = skip
        file = open(file_dir+'/data/train_lst.txt', 'r')
        for line in file.read().splitlines():
            data = line.split('_')
            self.video_list.append(file_dir+'/data/data_aligned/'+line)
            self.label_list.append(alpha2Index(data[1]))

        self.size = len(self.label_list)


    def __getitem__(self, index):

        folder = self.video_list[index]

        total_images = sorted(os.listdir(folder))

        imgs = []

        for i,img_name in enumerate(total_images[:200]):
            if i % 5 == self.skipframe:
                img = cv2.imread(f'{folder}/{img_name}')
                img = cv2.resize(img,(64,64))
                imgs.append(img)


        for i in range(len(imgs),40):
            imgs.append(np.zeros([64,64,3]))

        imgs = np.float32(imgs)/127.5-1
        imgs = imgs.flatten()
        label = self.label_list[index]


        return imgs , label

    def __len__(self):
        return self.size




class TestData(Data.Dataset):

    def __init__(self, file_dir,skip=0):
        super(TestData, self).__init__()

        self.video_list = []
        self.label_list = []
        self.skipframe = skip
        file = open(file_dir+'/data/eval_lst.txt', 'r')
        for line in file.read().splitlines():
            data = line.split('_')
            self.video_list.append(file_dir+'/data/data_aligned/'+line)
            self.label_list.append(alpha2Index(data[1]))

        self.size = len(self.label_list)


    def __getitem__(self, index):

        folder = self.video_list[index]

        total_images = sorted(os.listdir(folder))

        imgs = []

        for i,img_name in enumerate(total_images[:200]):
            if i % 5 == self.skipframe:
                img = cv2.imread(f'{folder}/{img_name}')
                img = cv2.resize(img,(64,64))
                imgs.append(img)


        for i in range(len(imgs),40):
            imgs.append(np.zeros([64,64,3]))

        imgs = np.float32(imgs)/127.5-1
        imgs = imgs.flatten()
        label = self.label_list[index]


        return imgs , label

    def __len__(self):
        return self.size





if __name__ == '__main__':


    DATASET = TrainData('./')
    DATALOADER = Data.DataLoader(
        DATASET,
        batch_size=5,
        shuffle=True,
        num_workers=4
        )
    for batch_idx, (imgs, label) in enumerate(DATALOADER):
        print(imgs.shape)
        print(label)