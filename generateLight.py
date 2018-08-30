import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

class LightGen(object):
    """
    generate lights which uniformly attribute in a unit sphere
    """
    # we have two methods for generate light : regular and random
    def __init__(self, N,method="random"):
        assert N,"Light number is zero"
        self.method=method
        self.N=N
        self.LightNum=0
        self.L=[]
        self.x=[]
        self.y=[]
        self.z=[]
        self.fi=(np.sqrt(5)-1)/2
        self.isHalfRequired=True


    def generate(self):

        if self.method=="regular":
            L=self.genRegular()
        elif self.method=="random":
            L =self.genRandom()
        else:
            raise RuntimeError('No specific method : regular/random, input is '.join(self.method))

        if self.isHalfRequired:
            # use z as index
            index=L[2]>0
            self.L = L[:,index]
        else:
            self.L = L

        _, self.LightNum=self.L.shape
        return self.L.transpose() #shape (N,3)




    def genRegular(self):
        index = np.arange(1, self.N + 1)
        self.z=(2*index-1)/self.N-1
        self.x=np.sqrt(1-self.z*self.z)*np.cos(2*math.pi*index*self.fi)
        self.y=np.sqrt(1-self.z*self.z)*np.sin(2*math.pi*index*self.fi)

        return np.vstack((self.x,self.y,self.z))

    def genRandom(self):
        np.random.seed(128)
        u = np.random.random(size=self.N)
        np.random.seed(256)
        v = np.random.random(size=self.N)
        theta = 2 * math.pi * u
        phi = np.arccos(2 * v - 1)
        self.x = np.sin(theta) * np.sin(phi)
        self.y = np.cos(theta) * np.sin(phi)
        self.z = np.cos(phi)
        return np.vstack((self.x, self.y, self.z))

    def drawLight(self):
        ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
        #  将数据点分成三部分画，在颜色上有区分度
        ax.scatter(self.L[0], self.L[1], self.L[2], c='b')  # 绘制数据点
        ax.set_zlabel('Z')  # 坐标轴
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        plt.show()






if __name__ == '__main__':

    generator=LightGen(2000,"regular")
    generator.generate()
    L=generator.L
    generator.drawLight()
    print('light shape is:'+' '.join(str(v) for v in L.shape))