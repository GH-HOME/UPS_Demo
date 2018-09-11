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


def fc(batchNorm, in_planes, out_planes):
    if batchNorm:
        return nn.Sequential(
            nn.Linear(in_planes,out_planes),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Linear(in_planes, out_planes),
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

    def __init__(self,batchNorm=True, input_Node=10*8*8,output_node=10):
        super(Upsnets,self).__init__()

        self.batchNorm=batchNorm
        self.input_Node=input_Node

        #non share weight
        self.fc1 = fc(self.batchNorm, input_Node, 32*7*7)
        self.fc2 = fc(self.batchNorm, 32*7*7, 64*5*5)
        self.fc3 = fc(self.batchNorm, 64*5*5, 128*3*3)
        self.fc4 = fc(self.batchNorm, 128*3*3, 128*3*3)
        self.fc5 = fc(self.batchNorm, 128*3*3, 256*3*3)
        self.fc6 = fc(self.batchNorm, 256 * 3 * 3, 128 * 3 * 3)

        self.fc_l1=nn.Linear(128*3*3,64*3)
        self.fc_l2 = nn.Linear(64*3, output_node*3)

        self.dropout = nn.Dropout(0.5)
        self.relu=nn.ReLU(inplace=True)
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
        images_flat=images.view(Batch_size,-1)

        assert images_flat.shape[1]==self.input_Node

        out_fc1=self.fc1(images_flat.float())
        out_fc2 = self.fc2(out_fc1)
        out_fc3=self.fc3(out_fc2)
        out_fc4=self.fc4(out_fc3)
        out_fc5=self.fc5(out_fc4)
        out_fc6=self.fc6(out_fc5)

        out_L1 = self.fc_l1(out_fc6)
        out_L1=self.relu(out_L1)
        out_L = self.fc_l2(out_L1)
        out_L = out_L.view(Batch_size, -1, 3)


        # out_L = out_L.squeeze()
        out_L_norm = torch.norm(out_L, 2, 2)

        out_L_norm = out_L_norm.view(-1).unsqueeze(1)
        out_L_norm = out_L_norm.repeat(1, 3)

        out_L = out_L.view(-1, 3)
        out_L = out_L / out_L_norm
        out_L = out_L.view(Batch_size, -1, 3)

        if train_writer is not None and printflag:
            train_writer.add_image('Internel/images', images.cpu().numpy()[0, 0, :, :])
            train_writer.add_image('Internel/out_fc1', out_fc1.detach().cpu().numpy()[0].reshape(32,-1))
            train_writer.add_image('Internel/out_fc4', out_fc4.detach().cpu().numpy()[0].reshape(128,-1))
            train_writer.add_image('Internel/out_fc6', out_fc6.detach().cpu().numpy()[0].reshape(128,-1))

        return out_L

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def upsnets(data=None,input_N=50,imagesize=8):
    model=Upsnets(batchNorm=False,input_Node=input_N*imagesize*imagesize,output_node=input_N)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def upsnets_bn(data=None,input_N=50, imagesize=8):

    model = Upsnets(batchNorm=True,input_Node=input_N*imagesize*imagesize,output_node=input_N)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def num_flat_features(x):
    size = x.size()[1:]  # 这里为什么要使用[1:],是因为pytorch只接受批输入，也就是说一次性输入好几张图片，那么输入数据张量的维度自然上升到了4维。【1:】让我们把注意力放在后3维上面
    num_features = 1
    for s in size:
        num_features *= s
    return num_features

