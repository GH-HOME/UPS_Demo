import glob
from .listdataset import ListDataset
from .util import split2list
import random
import cv2
import numpy as np
import os
import tqdm


def make_dataset(dir, split=None, lightSampleNum=50, ChoiseTime=100):
    images = []

    light_path=glob.glob(os.path.join(dir,'*gt.txt'))
    Light=np.loadtxt(light_path[0])
    ShapeSet=os.listdir(dir)
    index=range(Light.shape[0])
    print('=> make dataset for {} shape by {} times\n'.format(len(ShapeSet),ChoiseTime))
    for sdir in tqdm.tqdm(ShapeSet):
        path=os.path.join(dir,sdir)
        if os.path.isdir(path):
            image_list=sorted(glob.glob(os.path.join(path,'*_L_*.png')))
            if(len(image_list) != Light.shape[0]):
                raise RuntimeError("light num is not equal to rendered image num")

            for i in range(ChoiseTime):
                index_sample=random.sample(index,lightSampleNum)
                subImagepath=[]
                Light_sample = []
                [(subImagepath.append(image_list[j]), Light_sample.append(Light[j])) for j in index_sample]
                images.append([subImagepath,Light_sample])

    return split2list(images, split, default_split=0.9)


def Lambertian_direction(root, transform=None, target_transform=None,
                  co_transform=None, split=None,light_num=50):
    train_list, test_list = make_dataset(root,split,lightSampleNum=light_num)
    train_dataset = ListDataset(root, train_list, transform, target_transform, co_transform)
    test_dataset = ListDataset(root, test_list, transform, target_transform)

    return train_dataset, test_dataset
