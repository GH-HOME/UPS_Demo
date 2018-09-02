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
    def __init__(self, I):
        self.freedom = 3
        self.P_L = []
        self.P_N = []
        self.I=I
        self.imageSet=[]
        self.index=[]
        self.mask=[]

    def __init__(self, imagelist, mask_path):
        if imagelist is None or mask_path is None:
            raise RuntimeError("Empty image/mask list input")

        mask=cv2.imread(mask_path[0],0)
        self.shape = mask.shape
        self.mask = mask/255
        mask=mask.reshape(-1)
        self.imageSet = []
        self.index=np.where(mask!=0)

        I=[]
        for i in range(len(imagelist)):
            Ii=cv2.imread(imagelist[i],0)
            Ii=Ii/255
            self.imageSet.append(Ii)
            Ii=Ii.reshape(-1)
            I.append(Ii[self.index])

        #print("The shape of I is {}".format(len(I)))
        self.I=np.array(I).transpose()
        self.freedom = 3
        self.P_L = []
        self.P_N = []


    def calculate(self):


        start = time.time()
        U, sigma, VT = np.linalg.svd(self.I,full_matrices=False)
        duration=time.time()-start

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
        for i in range(len(self.index)):
            P_N_all[self.index[i]] = self.P_N[i]

        self.P_N=P_N_all.reshape(m,n,3)


        # print("P_N shape {} sigma shape {} P_L shape{}  during time is {}".format(self.P_N.shape, sigmaM.shape,
        #                                                                       self.P_L.shape, duration))

if __name__ == '__main__':

    path='./time_08_30_10_41_Light_30_shape_10_albedo_1/UPSDataset/blob01'
    Im_path_list=glob.glob(os.path.join(path,'blob01_L_*.png'))
    mask_path='./blobs/data/blob01_m.png'
    Pcalculator=Pseudo_NL_calculator(Im_path_list, mask_path)
    Pcalculator.calculate()

