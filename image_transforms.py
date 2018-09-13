from __future__ import division
import torch
import random
import numpy as np
import numbers
import types
import scipy.ndimage as ndimage

'''Set of tranform random routines that takes both input and target as arguments,
in order to have random but coherent transformations.
inputs are PIL Image pairs and targets are ndarrays'''


class Compose(object):
    """ Composes several co_transforms together.
    For example:
    >>> co_transforms.Compose([
    >>>     co_transforms.CenterCrop(10),
    >>>     co_transforms.ToTensor(),
    >>>  ])
    """

    def __init__(self, co_transforms):
        self.co_transforms = co_transforms

    def __call__(self, input, target):
        for t in self.co_transforms:
            input,target = t(input,target)
        return input,target


class ArrayToTensor(object):

    def __call__(self, inputs, targets):
        inputs['Imgs'] = torch.from_numpy(np.array(inputs['Imgs']))
        inputs['mask'] = torch.from_numpy(np.array(inputs['mask']))

        targets['light']=torch.from_numpy(np.array(targets['light']))

        return inputs, targets


class Lambda(object):
    """Applies a lambda as a transform"""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, input,target):
        return self.lambd(input,target)


class CenterCrop(object):
    """Crops the given inputs and target arrays at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    Careful, img1 and img2 may not be the same size
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, inputs, targets):
        _, h1, w1= inputs['Imgs'].shape
        th, tw = self.size
        x1 = int(round((w1 - tw) / 2.))
        y1 = int(round((h1 - th) / 2.))

        inputs['Imgs'] = inputs['Imgs'][:, y1: y1 + th, x1: x1 + tw]
        return inputs, targets


class Scale(object):
    """ Rescales the inputs and target arrays to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation order: Default: 2 (bilinear)
    """

    def __init__(self, size, order=2):
        self.size = size
        self.order = order

    def __call__(self, inputs, target):
        h, w, _ = inputs.shape
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return inputs,target
        if w < h:
            ratio = self.size/w
        else:
            ratio = self.size/h

        inputs = ndimage.interpolation.zoom(inputs, ratio, order=self.order)

        target = ndimage.interpolation.zoom(target, ratio, order=self.order)
        return inputs, target


# class RandomCrop(object):
#     """Crops the given PIL.Image at a random location to have a region of
#     the given size. size can be a tuple (target_height, target_width)
#     or an integer, in which case the target will be of a square shape (size, size)
#     """
#
#     def __init__(self, size):
#         if isinstance(size, numbers.Number):
#             self.size = (int(size), int(size))
#         else:
#             self.size = size
#
#     def __call__(self, inputs,target):
#         _, h, w = inputs['Imgs'].shape
#         th, tw = self.size
#         if w == tw and h == th:
#             return inputs,target
#
#         x_crop=0
#         y_crop=0
#         while True:
#             x1 = random.randint(0, w - tw)
#             y1 = random.randint(0, h - th)
#             patch=inputs['mask'][y1: y1 + th,x1: x1 + tw]
#             count=torch.nonzero(patch).size(0)
#             if count/(self.size[0]*self.size[1])>0.95:
#                 x_crop=x1
#                 y_crop=y1
#                 break
#
#         inputs['Imgs'] = inputs['Imgs'][:, y_crop: y_crop + th, x_crop: x_crop + tw]
#         return inputs, target


class RandomCrop(object):
    """Crops the given PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size, miniLight, ratio):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
            self.total_batch_pixel=size*size
            self.miniLight=miniLight
            self.ratio=ratio
        else:
            self.size = size
            self.total_batch_pixel = size[0] * size[1]
            self.miniLight = miniLight
            self.ratio = ratio

    def __call__(self, inputs,target):
        n, h, w = inputs['Imgs'].shape

        th, tw = self.size
        if w == tw and h == th:
            return inputs,target

        x_crop=0
        y_crop=0
        confidence=np.zeros(n)
        while True:
            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)
            patch=inputs['mask'][y1: y1 + th,x1: x1 + tw]
            count=torch.nonzero(patch).size(0)
            if count/(self.size[0]*self.size[1])>0.95:
                x_crop=x1
                y_crop=y1
                cropImage=inputs['Imgs'][:, y_crop: y_crop + th, x_crop: x_crop + tw]
                for i in range(n):
                    nonzero_num=torch.nonzero(cropImage[i]).size(0)
                    if nonzero_num/self.total_batch_pixel<self.ratio:
                        confidence[i]=0.0
                    else:
                        confidence[i] =(nonzero_num/self.total_batch_pixel)

                if np.nonzero(confidence)[0].shape[0]>self.miniLight:
                    break

        index=np.where(confidence!=0)
        inputs['Imgs'] = inputs['Imgs'][index, y_crop: y_crop + th, x_crop: x_crop + tw]
        target['light']=target['light'][index]

        confidence=torch.from_numpy(confidence)

        inputs['confidence']=confidence[index]
        return inputs, target


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, inputs, target):
        if random.random() < 0.5:
            n, w, h = inputs['Imgs'].shape
            for i in range(n):
                inputs['Imgs'][i]=torch.from_numpy(np.copy(np.fliplr(inputs['Imgs'][i])))

        return inputs,target


class RandomVerticalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """

    def __call__(self, inputs, target):
        if random.random() < 0.5:
            n,w,h = inputs['Imgs'].shape
            for i in range(n):
                inputs['Imgs'][i]=torch.from_numpy(np.copy(np.flipud(inputs['Imgs'][i])))

        return inputs,target


class RandomTranslate(object):
    def __init__(self, translation):
        if isinstance(translation, numbers.Number):
            self.translation = (int(translation), int(translation))
        else:
            self.translation = translation

    def __call__(self, inputs,target):
        h, w, _ = inputs.shape
        th, tw = self.translation
        tw = random.randint(-tw, tw)
        th = random.randint(-th, th)
        if tw == 0 and th == 0:
            return inputs, target
        # compute x1,x2,y1,y2 for img1 and target, and x3,x4,y3,y4 for img2
        x1,x2,x3,x4 = max(0,tw), min(w+tw,w), max(0,-tw), min(w-tw,w)
        y1,y2,y3,y4 = max(0,th), min(h+th,h), max(0,-th), min(h-th,h)

        inputs = inputs[y1:y2,x1:x2]

        target = target[y1:y2,x1:x2]


        return inputs, target



