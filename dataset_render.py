from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from shutil import copyfile
import glob
import os

import cv2
import numpy as np
import tqdm
import datetime
import matplotlib.pyplot as plt
from generateLight import LightGen
import mydraw
from CreateIntensityContour import *

class UPSDataset(object):
    '''
    Function: Read normal and mask from existing dataset and render the synthetic data
    '''

    def __init__(self, datasetName="UPSDataset",datarootdir="./dataset/",Nps=20):
        self.datarootdir=datarootdir
        self.datasetName=datasetName
        self.refdatasetname=['blobs/data'] # the existing dataset used to provide normal and mask image
        self.tardatasetdir=[] # dir for storing the synthetic data
        self.datalist=[]  # data list text which used to load data
        self.originalNlist=[]
        self.albedo=1
        self.total_Num = 500*2  # total light in the sphere, for the semi-sphere light, you need to *2
        self.isdrawContor=True


        self.Light=[]

    def load_Normal_Mask_from_dataset(self,datasetname):
        data_dir=os.path.join(self.datarootdir,datasetname)
        normal_img_list = sorted(glob.glob(os.path.join(data_dir, "*_n.png")))
        shape_num = len(normal_img_list)
        img_size = cv2.imread(normal_img_list[0])[:, :, 0].shape

        print("shape num in dataset {}: {}".format(datasetname, shape_num))
        print("image_size in dataset {}: {}".format(datasetname, img_size))

        N_list = []
        mask_list = []
        Sname_list=[]
        for n_path in normal_img_list:
            N, mask = self.__load_normal_png(n_path)
            N_list.append(N)
            mask_list.append(mask)
            shape_name=os.path.basename(n_path)
            Sname_list.append(shape_name[:-6])

        self.originalNlist=N_list
        return shape_num, Sname_list, img_size, N_list, mask_list




    def __load_normal_png(self, n_path):
        n_img = cv2.imread(n_path)[:, :, ::-1]
        m, n, _ = n_img.shape

        N = n_img.reshape(-1, 3).T
        N = N.astype(np.float32) / 255. * 2. - 1.
        for i in range(m * n):
            norm = np.linalg.norm(N[:, i])
            if norm != 0:
                N[:, i] /= norm

        mask = np.ones(shape=(m * n))
        n_img = n_img.reshape(-1, 3).T
        for i in range(m * n):
            if np.linalg.norm(n_img[:, i]) == 0:
                mask[i] = 0

        N[:, mask == 0] = 0
        return N, mask

    def render_calc(self, N, img_size, mask, L):
        """

        :param np.ndarray N: (3, m * n)
        :param np.ndarray mask: (m * n)
        :param np.ndarray L: (light_num, 3)
        :type func_brdf: (light, normal, view) -> (3, )
        :return: Measurement matrix (light_num, 3, m * n)
        :rtype: np.ndarray
        """
        m, n=img_size
        light_num, _ = L.shape
        M = np.zeros(shape=(light_num, 3, m * n))
        view_vec = (0, 0, 1)

        for l in range(light_num):
            for i in range(m * n):
                if mask[i] == 0:
                    continue

                normal = np.array(N[:, i]).flatten()
                normal /= np.linalg.norm(normal)
                light = np.array(L[l, :]).flatten()
                light /= np.linalg.norm(light)

                rhos = self.albedo
                nl = float(np.dot(light, normal))
                nl = max(0., nl)
                M[l, :, i] = np.array(rhos) * nl

        return M

    def output(self, M, N, L, img_size, mask, output_path, obj_name):

        m, n = img_size
        self.tardatasetdir = os.path.join(output_path, self.datasetName, obj_name)
        if not os.path.exists(self.tardatasetdir):
            os.makedirs(self.tardatasetdir)

        light_num, _ = L.shape

        assert M.shape == (light_num, 3, m * n)
        assert N.shape == (3, m * n)
        mask_file_name= obj_name+"_M.png"
        mask_file_name = os.path.join(self.tardatasetdir, mask_file_name)

        normal_file_name = obj_name + "_N.txt"
        normal_file_name = os.path.join(self.tardatasetdir, normal_file_name)
        np.savetxt(normal_file_name, N)

        cv2.imwrite(mask_file_name, mask.reshape(img_size)*255)
        for l in range(light_num):
            m_img = M[l, :, :]  # [light_num, 3, m*n]
            m_img = m_img.astype(np.float) / np.max(M) * np.iinfo(np.uint16).max
            m_img = m_img.T.reshape(m, n, 3).astype(np.uint16)
            write_path=os.path.join(self.tardatasetdir, obj_name+"_L_{:0>4d}.png".format(l))
            self.datalist.append(write_path)

            # cv2.imshow('',m_img[:, :, ::-1])
            # cv2.waitKey()
            cv2.imwrite(write_path, m_img[:, :, ::-1])
            if self.isdrawContor:
                createContor = CreateIntensityContour(write_path, 12)
                createContor.createContour()



    def rendering(self):

        # load light
        L_generator = LightGen(N=self.total_Num, method="regular")
        self.Light=L_generator.generate()


        # load normal and mask
        shape_num, Sname_list, img_size, N_list, mask_list = self.load_Normal_Mask_from_dataset(self.refdatasetname[0])
        timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M")
        data_info = "time_{}_Light_{}_shape_{}_albedo_{}".format(timestamp, L_generator.LightNum, shape_num, self.albedo)
        output_path=os.path.join(self.datarootdir,data_info)

        for i in tqdm.tqdm(range(shape_num)):
            M=self.render_calc(N_list[i],img_size,mask_list[i],self.Light)
            self.output(M, N_list[i],self.Light, img_size, mask_list[i], output_path, Sname_list[i])


        self.save_txt_dir = os.path.join(output_path,self.datasetName, "L_gt.txt")
        np.savetxt(self.save_txt_dir, self.Light)





def drawLight_distribute_from_txt(save_txt_dir, scale=200):
    #only for unit light
    dirpath=os.path.dirname(save_txt_dir)
    save_light_png_path=os.path.join(dirpath,"LightGT")

    if not os.path.exists(save_light_png_path):
        os.makedirs(save_light_png_path)

    Light=np.loadtxt(save_txt_dir)
    Light_t=Light.transpose()
    n,_=Light.shape
    for l in range(n):
        light_vec=Light[l]
        if np.abs(np.sqrt(np.abs(np.dot(light_vec,light_vec)-1))) > 1e-4:
            raise RuntimeError("No normalized light direction！")
        image=np.zeros([scale,scale],np.uint8)
        x = int((scale * (light_vec[0] + 1)/2))
        y = int((scale * (light_vec[1] + 1)/2))

        image[x,y]=255
        cv2.imwrite("{}/L{}.png".format(save_light_png_path,l),image)

        ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
        #  将数据点分成三部分画，在颜色上有区分度

        ax.scatter(light_vec[1],light_vec[0],light_vec[2], c='b')  # 绘制数据点
        ax.scatter(Light_t[1], Light_t[0], Light_t[2], c='g')  # 绘制数据点
        ax.set_zlabel('Z')  # 坐标轴
        ax.set_ylabel('X')
        ax.set_xlabel('Y')
        #plt.show()
        plt.savefig("{}/L3D_ref_{}.png".format(save_light_png_path, l))







if __name__ == '__main__':

    upsdata=UPSDataset()
    upsdata.rendering()
    #drawLight_distribute_from_txt(upsdata.save_txt_dir)








