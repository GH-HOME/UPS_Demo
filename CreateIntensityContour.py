import cv2
import numpy as np
import os

class CreateIntensityContour(object):

    def __init__(self,imgpath,intervel=8):
        self.image=cv2.imread(imgpath,0)
        self.intervel=intervel
        self.dirpath=os.path.dirname(imgpath)
        self.dirname=os.path.basename(imgpath)[:-4]
        if self.image is None:
            raise RuntimeError('image is empty of path {}'.format(imgpath))

    def createContour(self):
        index=np.arange(255,1,-self.intervel)

        dirpath=os.path.join(self.dirpath,self.dirname)
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)

        for i in range(index.shape[0]-1):
            name=self.dirname[:-4]+'{}_{}.png'.format(index[i],index[i+1])
            path=os.path.join(dirpath,name)
            chooseid = np.where((self.image < index[i]) & (self.image > index[i + 1]))
            result=np.zeros_like(self.image,np.uint8)
            result[chooseid]=255
            cv2.imwrite(path,result)



if __name__ == '__main__':
    imagepath='./dataset/time_09_02_11_55_Light_500_shape_10_albedo_1/UPSDataset/blob01/blob01_L_0045.png'
    createContor=CreateIntensityContour(imagepath,12)
    createContor.createContour()



