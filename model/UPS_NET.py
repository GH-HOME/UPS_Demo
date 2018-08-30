import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal

__all__ =[

    'upsnets', 'upsnets_bn'
]


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes,out_planes,kernel_size=kernel_size, stride=stride,padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                      bias=False),
            nn.ReLU(inplace=True)
        )

def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes,out_planes,kernel_size=4, stride=2,padding=1,bias=False),
        nn.ReLU(inplace=True)
    )


def crop_like(input, target):
    if input.size()[2:]  == target.size()[2:]:
        return  input
    else:
        return input[:,:,:target.size(2),:target.size(3)]


class Upsnets(nn.Module):

    def __init__(self,batchNorm=True,input_N=20):
        super(Upsnets,self).__init__()

        self.batchNorm=batchNorm

        self.conv1=conv(self.batchNorm, input_N, 64,kernel_size=7,stride=2)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=3, stride=2)
        self.conv4 = conv(self.batchNorm, 256, 256, kernel_size=3, stride=1)


        # for the normal line
        self.deconv1=deconv(256,128)
        self.deconv2 = deconv(128, 64)

        self.convN_1=conv(False,1,1,kernel_size=1,stride=1)

        self.fc1=nn.Linear(192*192,1024)
        self.fc2=nn.linear(1024,9)

        # for the light line
        self.convL_1=conv(self.batchNorm,256,128,kernel_size=3,stride=1)
        self.convL_2=conv(False,128,1,kernel_size=3,stride=1)

        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
                elif isinstance(m,nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

        def forward(self, x):
            # here x is a P*P*C matrix normalized to (0-1) C is the light number, P is the shape
            # the data structure is [N C H W]

            # we first calculate the Pseudo normal and light

            batch_num, light_num, height, width=x.shape

            with torch.no_grad():
                for i in range(batch_num):
                    I=x[i]
                    print(I.shape)


        def weight_parameters(self):
            return [param for name, param in self.named_parameters() if 'weight' in name]

        def bias_parameters(self):
            return [param for name, param in self.named_parameters() if 'bias' in name]


def upsnets(data=None):
    model=Upsnets(batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def upsnets_bn(data=None):
    model = Upsnets(batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model




