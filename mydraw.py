from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations




# draw a vector
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):

    def __init__(self, point2, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        point2=point2.transpose()

        self._verts3d = point2[0], point2[1], point2[2]


    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def drawlightseq(Light):

    fig = plt.figure(figsize=(25.,25.))
    ax = fig.gca(projection='3d')
    ax.set_aspect("equal")

    # draw cube
    # r = [-1, 1]
    # for s, e in combinations(np.array(list(product(r, r, r))), 2):
    #     if np.sum(np.abs(s-e)) == r[1]-r[0]:
    #         ax.plot3D(*zip(s, e), color="b")

    # draw sphere
    u, v = np.mgrid[0:2 * np.pi:25j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="r")

    # draw a point
    ax.scatter([0], [0], [0], color="g", s=100)

    n, _ = Light.shape
    #change view direction
    ax.view_init(elev=90,azim=0)
    for i in range(n):
        lightvector=np.zeros([2,3],np.float)
        lightvector[1]=Light[i]
        a = Arrow3D(lightvector, mutation_scale=5,
                    lw=3, arrowstyle="-|>", color="k")
        ax.add_artist(a)


    plt.show()


if __name__ == '__main__':
    X = np.zeros([2,3],np.float)
    X[1,0]=0.15519809
    X[1,1]=0.09810481
    X[1,2]=0.983
    X[0, 0] = -0.07333659
    X[0, 1] = -0.26413813
    X[0, 2] = 0.96169275

    drawlightseq(X)