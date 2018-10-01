import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal
import numpy as np

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


def fc(in_planes, out_planes):
    return nn.Sequential(
        nn.Linear(in_planes,out_planes),
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

    def __init__(self,batchNorm=True, input_size=128,LightNum=10):
        super(Upsnets,self).__init__()

        self.batchNorm=batchNorm
        self.input_size=input_size
        self.LightNum=LightNum

        #non share weight

        self.conv1 = conv(self.batchNorm, 1, 32, kernel_size=5, stride=1)
        self.conv2 = conv(self.batchNorm, 32, 64, kernel_size=3, stride=1)
        self.conv3 = conv(self.batchNorm, 64, 128, kernel_size=3, stride=1)
        self.conv4 = conv(self.batchNorm, 128, 256, kernel_size=3, stride=1)
        self.conv5 = conv(self.batchNorm, 256, 3, kernel_size=3, stride=1)

        self.encoder=nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5
        )

        self.conv_image_light1=conv(self.batchNorm,1+3,32,kernel_size=3,stride=1)
        self.conv_image_light2 = conv(self.batchNorm, 32, 1, kernel_size=3, stride=1)
        self.fc1 = fc(input_size*input_size, 32*5)
        self.fc2 = fc(32*5, 64*3)
        self.fc_l2 = nn.Linear(64*3, 3)
        self.dropout = nn.Dropout(0.5)

        self.feature_compress = nn.Sequential(
            self.conv_image_light1,
            self.conv_image_light2
        )

        self.light_infer=nn.Sequential(
            self.fc1,
            #self.dropout,
            self.fc2,
            self.dropout,
            self.fc_l2
        )
        self.relu=nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d) or isinstance(m,nn.Linear):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.out_encoder_max_single = np.load('out_encoder_max.npy')

    def forward(self, inputs,train_writer=None, printflag=False):
        images = inputs['Imgs']
        masks=inputs['mask']
        Batch_size, Light_num, w,h =images.shape

        out_encoder = torch.randn((Light_num, Batch_size, 3, w, h), dtype=torch.float).cuda()
        for i in range(Light_num):
            input_image_single=images[:,i,:,:].unsqueeze(1).float()
            out_encoder[i]=self.encoder(input_image_single)

        # Max sample
        out_encoder_max=torch.max(out_encoder,0)[0]  # this can be used to weak supervise normal and albedo
        # Average sample
        #out_encoder_max = torch.mean(out_encoder, 0)  # this can be used to weak supervise normal and albedo
        out_encoder_max=torch.where(masks.unsqueeze(1)!=0, out_encoder_max,masks.unsqueeze(1).float())

        # out_encoder_max=torch.from_numpy(self.out_encoder_max_single.repeat(Batch_size,axis=0)).cuda()
        # out_encoder_max = torch.where(masks.unsqueeze(1) != 0, out_encoder_max, masks.unsqueeze(1).float())

        out_L=torch.randn((Batch_size, Light_num, 3), dtype=torch.float).cuda()

        for b in range(Batch_size):
            for i in range(Light_num):
                input_single_image = images[b, i, :, :].unsqueeze(0).float()
                out_concat=torch.cat((out_encoder_max[b],input_single_image),0).unsqueeze(0)
                feature_compress=self.feature_compress(out_concat)
                feature_compress=feature_compress.view(-1)
                out_L[b, i]=self.light_infer(feature_compress)


        # out_L = out_L.squeeze()
        out_L_norm = torch.norm(out_L, 2, 2)

        out_L_norm = out_L_norm.view(-1).unsqueeze(1)
        out_L_norm = out_L_norm.repeat(1, 3)

        out_L = out_L.view(-1, 3)
        out_L = out_L / out_L_norm
        out_L = out_L.view(Batch_size, -1, 3)

        if train_writer is not None and printflag:
            train_writer.add_image('Internel/images', images.cpu().numpy()[0, 0, :, :])
            train_writer.add_image('Internel/mask', masks.cpu().numpy()[0, :, :])
            train_writer.add_image('Internel/out_encoder_max', out_encoder_max.detach().cpu().numpy()[0])


        return out_L

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def upsnets(data=None,input_N=50,imagesize=128):
    model=Upsnets(batchNorm=False,input_size=imagesize,LightNum=input_N)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def upsnets_bn(data=None,input_N=50, imagesize=128):

    model = Upsnets(batchNorm=True,input_size=imagesize,LightNum=input_N)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def num_flat_features(x):
    size = x.size()[1:]  # 这里为什么要使用[1:],是因为pytorch只接受批输入，也就是说一次性输入好几张图片，那么输入数据张量的维度自然上升到了4维。【1:】让我们把注意力放在后3维上面
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

