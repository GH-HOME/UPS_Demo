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
        self.conv1 = conv(self.batchNorm, input_N, 64, kernel_size=7, stride=2)

        self.conv2 = conv(self.batchNorm, 64, 128, kernel_size=5, stride=1)
        self.conv3 = conv(self.batchNorm, 128, 256, kernel_size=3, stride=2)
        self.conv4 = conv(self.batchNorm, 256, 256, kernel_size=3, stride=1)
        self.pool = nn.MaxPool2d(2, 2)

        # for the normal line
        self.deconv1=deconv(256,128)
        self.deconv2 = deconv(128, 64)

        self.convN_1=conv(False,67,1,kernel_size=1,stride=1)

        self.fc1=nn.Linear(180*180,1024)
        self.fc2=nn.Linear(1024,9)

        # for the light line
        self.convL_1=conv(self.batchNorm,256,128,kernel_size=3,stride=1)
        # self.convL_2=conv(False,129,1,kernel_size=3,stride=1)
        self.convL_2 = conv(False, 128, 1, kernel_size=3, stride=1)

        for m in self.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
                elif isinstance(m,nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def forward(self, inputs):
        # here x is a P*P*C matrix normalized to (0-1) C is the light number, P is the shape
        # the data structure is [N C H W]

        # we first calculate the Pseudo normal and light
        images, P_L, P_N, mask=inputs['Imgs'], inputs['P_L'], inputs['P_N'], inputs['mask']
        P_L=torch.unsqueeze(P_L,1)
        #P_N=P_N.permute([3,0,1,2]) #这里维度顺序还是不对
        N, C, _, _ = images.shape
        # out_encoder=torch.zeros([N,C,256,32,32])
        # for i in range(C):
        #     #encoder part
        #     image_Ci=torch.unsqueeze(images[:,i,:,:],1).float()
        #     out_conv2 = self.conv2(self.conv1(image_Ci))
        #     out_conv4 = self.conv4(self.conv3(out_conv2))
        #     out_encoder[:,i,:,:,:].data=out_conv4.clone()
        # #for share weight and merge
        # out_conv4_cpu=torch.max(out_encoder,1)[0]
        # out_conv4 = out_conv4_cpu.type(torch.cuda.FloatTensor)

        out_conv2 = self.conv2(self.conv1(images.float()))
        out_conv4 = self.conv4(self.conv3(out_conv2))


        #light line
        out_deconv_L1=self.convL_1(out_conv4)

        # concat the pesudo light
        # out_deconv_L2=torch.cat([out_deconv_L1,P_L.float()],1)
        # out_L=self.convL_2(out_deconv_L2)

        # without concating pesudo light
        out_L = self.convL_2(out_deconv_L1)

        #normal line
        # out_deconv_N=self.deconv2(self.deconv1(out_conv4))
        # mask=torch.squeeze(mask)
        # for i in range(mask.shape[0]):
        #     for j in range(mask.shape[1]):
        #         if mask[i,j]==0:
        #             out_deconv_N[:,:,i,j]=0
        # print('4')
        # out_deconv_N=self.convN_1(torch.cat([out_deconv_N,P_N.float()],1))
        # print('5')
        # # estimate ambiguity matrix
        # out_deconv_N_flat=out_deconv_N.view(-1, num_flat_features(out_deconv_N))
        # print('6')
        # out_AM=self.fc2(self.fc1(out_deconv_N_flat))
        # print('7')

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

