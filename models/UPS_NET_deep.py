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

    def __init__(self,batchNorm=True,input_N=50):
        super(Upsnets,self).__init__()

        self.batchNorm=batchNorm
        self.input_light_num=input_N

        #share weight
        #self.conv1 = conv(self.batchNorm, 1, 64,kernel_size=7,stride=2)

        #non share weight
        self.conv0 = conv(self.batchNorm, input_N, 32, kernel_size=7, stride=1)
        self.conv1 = conv(self.batchNorm, 32, 64, kernel_size=7, stride=1)
        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=1)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=3, stride=1)
        self.conv3_1 = conv(self.batchNorm, 256, 256, kernel_size=3, stride=1)
        self.conv4 = conv(self.batchNorm, 256, 512, kernel_size=3, stride=1)
        self.conv4_1 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)
        self.conv5 = conv(self.batchNorm, 512, 1024, kernel_size=3, stride=1)
        self.conv5_1 = conv(self.batchNorm, 1024, 1024, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.convN_1 = conv(self.batchNorm, 1024, 3, kernel_size=1, stride=1)
        self.fc_l1=nn.Linear(3*8*8,128)
        self.fc_l2 = nn.Linear(128, input_N*3)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)

        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d) or isinstance(m,nn.Linear):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
                elif isinstance(m,nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, inputs,train_writer=None, printflag=False):
        images = inputs['Imgs']
        Batch_size, Light_num, w,h =images.shape
        out_conv0 = self.conv0(images.float())
        out_conv1=self.conv1(out_conv0)
        out_conv1_pool=self.pool(out_conv1)
        out_conv2 = self.conv2(out_conv1_pool)
        out_conv2_pool = self.pool(out_conv2)
        out_conv3=self.conv3(out_conv2_pool)
        out_conv3_1=self.conv3_1(out_conv3)
        out_conv3_pool = self.pool(out_conv3_1)
        out_conv4=self.conv4(out_conv3_pool)
        out_conv4_1=self.conv4_1(out_conv4)
        out_conv4_pool = self.pool(out_conv4_1)
        out_conv5 = self.conv5(out_conv4_pool)
        out_conv5_1 = self.conv5_1(out_conv5)
        out_conv_img = self.convN_1(out_conv5_1)
        out_conv_img_flat = out_conv_img.view(-1, num_flat_features(out_conv_img))
        out_L = self.fc_l2(self.dropout(self.relu(self.fc_l1(out_conv_img_flat))))
        out_L = out_L.view(Batch_size,-1,3)

        # out_L = out_L.squeeze()
        out_L_norm = torch.norm(out_L, 2, 2)

        out_L_norm = out_L_norm.view(-1).unsqueeze(1)
        out_L_norm = out_L_norm.repeat(1, 3)

        out_L = out_L.view(-1, 3)
        out_L = out_L / out_L_norm
        out_L = out_L.view(Batch_size, -1, 3)

        if train_writer is not None and printflag:
            train_writer.add_image('Internel/images', images.cpu().numpy()[0, 0, :, :])
            train_writer.add_image('Internel/out_conv0', out_conv0.detach().cpu().numpy()[0, 0, :, :])
            train_writer.add_image('Internel/out_conv3', out_conv3.detach().cpu().numpy()[0, 0, :, :])
            train_writer.add_image('Internel/out_conv_img', out_conv_img.detach().cpu().numpy()[0, 0, :, :])

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

