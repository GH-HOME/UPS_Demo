import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.fc2=nn.Linear(1024,9)

        # for the light line
        self.convL_1=conv(self.batchNorm,256,128,kernel_size=3,stride=1)
        self.convL_2=conv(False,129,1,kernel_size=3,stride=1)

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
        images, P_L, P_N, mask=x['Imgs'], x['P_L'], x['P_N'], x['mask']

        #build observe map for light
        scale_size=45
        P_observeMap=torch.zeros(scale_size,scale_size)
        listsize=list(P_L.shape)
        for i in range(listsize[0]):
            x = (P_L[i,0]*0.5+0.5)*scale_size
            y = (P_L[i, 1] * 0.5 + 0.5) * scale_size
            z = (P_L[i, 2] * 0.5 + 0.5) * scale_size
            P_observeMap[x,y]=z

        #encoder part
        out_conv2 = self.conv2(self.conv1(x))
        out_conv4 = self.conv3(self.conv3(out_conv2))

        #light line
        out_deconv_L1=self.convL_1(out_conv4)
        out_deconv_L2=torch.cat([out_deconv_L1,P_observeMap],1)
        out_L=self.convL_2(out_deconv_L2)

        #normal line
        out_deconv_N=self.deconv2(self.deconv1(out_conv4))
        listsize=list(mask)
        for i in range(listsize[0]):
            for j in range(listsize[1]):
                if mask[i,j]==0:
                    out_deconv_N[:,:,i,j]=0
        out_deconv_N=self.convN_1(torch.cat([out_deconv_N,P_N],1))
        # estimate ambiguity matrix
        out_deconv_N_flat=out_deconv_N.view(-1, self.num_flat_features(out_deconv_N))
        out_AM=self.fc2(self.fc1(out_deconv_N_flat))

        return out_L,out_AM

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




