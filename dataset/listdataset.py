import torch.utils.data as data
from scipy.ndimage import imread
import numpy as np
from .Calculate_Pseudo_NL import Pseudo_NL_calculator as P_Calc

def default_loader(path_imgs):
    return [imread(img).astype(np.float32) for img in path_imgs]


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
        subImagepath, Light_sample, mask_path, normal_path = self.path_list[index]

        P_NL=P_Calc(subImagepath, mask_path)
        P_NL.calculate()

        normal=np.loadtxt(normal_path)
        inputs={'Imgs':P_NL.imageSet, 'P_L':P_NL.P_L, 'P_N':P_NL.P_N, 'mask':P_NL.mask}
        targets={'light':Light_sample, 'normal':normal}


        if self.co_transform is not None:
            inputs, target = self.co_transform(inputs, target)
        if self.transform is not None:
            inputs, targets = self.transform(inputs, targets)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return inputs, targets

    def __len__(self):
        return len(self.path_list)
