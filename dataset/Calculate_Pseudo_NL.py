import numpy as np
import cv2
import glob
import os
import time
from sklearn.preprocessing import normalize



class Pseudo_NL_calculator(object):
    '''
    calculate the pseudo normal and light direction according to the SVD decomposition
    I is M*N matrix M is the number of pixel
    '''

    def __init__(self, image_input, mask_input):
        if image_input is None or mask_input is None:
            raise RuntimeError("Empty image/mask list input")

        if isinstance(mask_input, str):
            mask=cv2.imread(mask_input,0)
            self.shape = mask.shape
            self.mask = mask/255
            mask=mask.reshape(-1)
            self.index=np.where(mask!=0)
            I=[]
            for i in range(len(image_input)):
                Ii=cv2.imread(image_input[i],0)
                Ii=Ii/255
                Ii=Ii.reshape(-1)
                I.append(Ii[self.index])

            self.I=np.array(I).transpose()

        else:
            self.mask=mask_input
            self.shape = mask_input.shape
            mask = mask_input.reshape(-1)
            self.index = np.where(mask != 0)
            l, h, w = image_input.shape
            image_input=image_input.reshape(l,-1)
            image_input=image_input[:,self.index[0]]
            self.I = image_input.transpose()

        self.P_L = []
        self.P_N = []


    def calculate(self):

        U, sigma, VT = np.linalg.svd(self.I,full_matrices=False)

        self.P_N=U[:, 0:3]
        self.P_L=VT[0:3,:]
        sigma=sigma[0:3]

        sigmaM=np.diag(sigma)

        self.P_N=np.dot(self.P_N,np.sqrt(sigmaM))
        self.P_L=np.dot(np.sqrt(sigmaM),self.P_L).transpose()

        self.P_N = normalize(self.P_N, axis=1)
        self.P_L = normalize(self.P_L, axis=0)

        m,n=self.shape
        P_N_all=np.zeros((m*n,3),dtype=np.float)
        for i in range(len(self.index[0])):
            P_N_all[self.index[0][i]] = self.P_N[i]

        self.P_N=P_N_all.reshape(m,n,3)


if __name__ == '__main__':

    path='./time_10_03_22_46_Light_30_shape_10_albedo_1/UPSDataset/blob01'
    Im_path_list=glob.glob(os.path.join(path,'blob01_L_*.png'))
    mask_path= os.path.join(path,'blob01_M.png')
    Pcalculator=Pseudo_NL_calculator(Im_path_list, mask_path)
    Pcalculator.calculate()

