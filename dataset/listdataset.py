import torch.utils.data as data
from scipy.ndimage import imread
import numpy as np
from matplotlib import pyplot as plt
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


        w,h=P_NL.shape
        normal=np.loadtxt(normal_path[0]).transpose()
        normal=normal.reshape(w,h,3)
        # normal_all=np.zeros([w*h,3],np.float)
        # normal_all[P_NL.index[0]]=normal

        inputs={'Imgs':P_NL.imageSet, 'P_L':self.CreatObservemapFromL(P_NL.P_L), 'P_N':P_NL.P_N, 'mask':P_NL.mask}
        targets={'light':self.CreatObservemapFromL(Light_sample), 'normal':normal}
        # plt.imshow(inputs['P_L'],cmap=plt.cm.gray)
        # plt.imshow(targets['light'], cmap=plt.cm.gray)
        # plt.show()


        if self.co_transform is not None:
            inputs, targets = self.co_transform(inputs, targets)
        if self.transform is not None:
            inputs, targets = self.transform(inputs, targets)
        if self.target_transform is not None:
            targets = self.target_transform(targets)


        return inputs, targets

    def CreatObservemapFromL(self, L=None, scale_size = 32):
        # build observe map for light
        if L is None:
            raise RuntimeError("no light input")
        if type(L) == list:
            L=np.array(L)
        P_observeMap = np.zeros([scale_size, scale_size],np.float)
        for i in range(L.shape[0]):
            x = int((L[i, 0] * 0.5 + 0.5) * scale_size)
            y = int((L[i, 1] * 0.5 + 0.5) * scale_size)
            z = (L[i, 2] * 0.5 + 0.5) * scale_size
            P_observeMap[x, y] = z

        return P_observeMap

    def __len__(self):
        return len(self.path_list)
