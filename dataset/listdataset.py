import torch.utils.data as data
import cv2
import numpy as np
from matplotlib import pyplot as plt
from .Calculate_Pseudo_NL import Pseudo_NL_calculator as P_Calc


def default_loader(path_imgs):
    images=[]
    for img in path_imgs:
        I=cv2.imread(img,0)
        I=I/255
        images.append(I)
    return images


class ListDataset(data.Dataset):
    def __init__(self, root, path_list, transform=None, target_transform=None,
                 co_transform=None, loader=default_loader):

        self.root = root
        self.path_list = path_list
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.loader = loader

    def __getitem__(self, index):
        subImagepath, Light_sample, mask_path = self.path_list[index]
        images=self.loader(subImagepath)
        mask=cv2.imread(mask_path[0],0)/255
        inputs={'Imgs':images, 'mask':mask}
        targets={'light':Light_sample}


        if self.co_transform is not None:
            inputs, targets = self.co_transform(inputs, targets)
        if self.transform is not None:
            inputs, targets = self.transform(inputs, targets)
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return inputs, targets



    def __len__(self):
        return len(self.path_list)
