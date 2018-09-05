import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal

__all__ =[

    'upsnets', 'upsnets_bn'
]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                  padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                  padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.inplanes=inplanes
        self.planes=planes
        self.downsample= nn.Conv2d(self.inplanes, self.planes * self.expansion,
                      kernel_size=1, stride=self.stride, bias=False)
        self.downsamplebn=nn.BatchNorm2d(self.planes * self.expansion)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.stride != 1 or self.inplanes != self.planes * self.expansion:
            residual=self.downsample(x)
            out=self.downsamplebn(out)

        out += residual
        out = self.relu(out)

        return out



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

    def __init__(self,batchNorm=True,input_N=50):
        super(Upsnets,self).__init__()

        self.batchNorm=batchNorm
        self.input_light_num=input_N

        #share weight
        #self.conv1 = conv(self.batchNorm, 1, 64,kernel_size=7,stride=2)

        #non share weight
        self.conv0 = conv(self.batchNorm, input_N, 64, kernel_size=7, stride=1)

        self.res0 = BasicBlock(64, 64, stride=1)
        self.res1 = BasicBlock(64,128,stride=1)
        self.res2 = BasicBlock(128, 256, stride=1)
        self.res3 = BasicBlock(256, 512, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.convN_1 = conv(False, 512, 3, kernel_size=1, stride=1)
        self.fc_l1=nn.Linear(3*16*16,16)
        self.fc_l2 = nn.Linear(16, input_N*3)
        self.avgpool = nn.AvgPool2d(512, stride=1)

        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
                elif isinstance(m,nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, inputs,train_writer=None, showindex=1,show_intervel=50):
        images = inputs['Imgs']
        Batch_size, Light_num, w,h =images.shape
        out_conv0 = self.pool(self.conv0(images.float()))
        out_res0 = self.res0(out_conv0)
        out_res1 = self.pool(self.res1(out_res0))
        out_res2 = self.res2(out_res1)
        out_res3 = self.pool(self.res3(out_res2))

        out_conv_img = self.convN_1(out_res3)
        out_conv_img_flat = out_conv_img.view(-1, num_flat_features(out_conv_img))
        out_L = self.fc_l2(self.fc_l1(out_conv_img_flat))
        out_L = out_L.view(Batch_size,-1,3)

        #out_L = out_L.squeeze()
        out_L_norm = torch.norm(out_L, 2, 2)

        out_L_norm = out_L_norm.view(-1).unsqueeze(1)
        out_L_norm = out_L_norm.repeat(1, 3)

        out_L = out_L.view(-1, 3)
        out_L = out_L / out_L_norm
        out_L = out_L.view(Batch_size, -1, 3)

        if train_writer is not None and showindex % show_intervel==0:
            train_writer.add_image('Internel/out_conv0', out_conv0.detach().cpu().numpy()[0,0,:,:],showindex)
            train_writer.add_image('Internel/out_res1', out_res1.detach().cpu().numpy()[0,0,:,:],showindex)
            train_writer.add_image('Internel/out_res3', out_res3.detach().cpu().numpy()[0,0,:,:],showindex)
            train_writer.add_image('Internel/out_conv_img', out_conv_img.detach().cpu().numpy()[0, 0, :, :], showindex)

        return out_L

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def upsnets(data=None,input_N=50):
    model=Upsnets(batchNorm=False,input_N=input_N)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def upsnets_bn(data=None,input_N=50):
    model = Upsnets(batchNorm=True,input_N=input_N)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def num_flat_features(x):
    size = x.size()[1:]  # 这里为什么要使用[1:],是因为pytorch只接受批输入，也就是说一次性输入好几张图片，那么输入数据张量的维度自然上升到了4维。【1:】让我们把注意力放在后3维上面
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

