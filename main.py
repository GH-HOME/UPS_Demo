import argparse
import  os
import shutil
import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as tramsforms
import image_transforms
import model
import dataset

import datetime
from tensorboardX import SummaryWriter
import numpy as np

model_names=sorted(name for name in model.__dict__
                   if name.islower() and not name.startswith("__"))


parser = argparse.ArgumentParser(description='PyTorch UPSNet Training on several Datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--datadir', metavar='DIR',default='./dataset/time_08_30_08_24_Light_500_shape_10_albedo_1/UPSDataset',
                    help='path to dataset')
parser.add_argument('--dataname', metavar='DataName',default='Lambertian_direction',
                    help='data set name')
group = parser.add_mutually_exclusive_group()
group.add_argument('-s', '--split-file', default=None, type=str,
                   help='test-val split file')
group.add_argument('--split-value', default=0.8, type=float,
                   help='test-val split proportion (between 0 (only test) and 1 (only train))')

parser.add_argument('--arch', '-a', metavar='ARCH', default='upsnets',
                    choices=model_names,
                    help='model architecture, overwritten if pretrained is specified: ' +
                    ' | '.join(model_names))
parser.add_argument('--solver', default='adam',choices=['adam','sgd'],
                    help='solver algorithms')
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch-size', default=1000, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                    help='beta parameter for adam')
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float,
                    metavar='W', help='weight decay')
parser.add_argument('--bias-decay', default=0, type=float,
                    metavar='B', help='bias decay')
parser.add_argument('--no_date',default=False,type=bool,help='If use data in folder name')
parser.add_argument('--pretrained', dest='pretrained', default=None,
                    help='path to pre-trained model')

best_EPE = -1
n_iter = 0

def main():
    global args, best_EPE, save_path
    args=parser.parse_args()
    save_path='{},{},{}epochs{},b{},lr{}'.format(
        args.arch,
        args.solver,
        args.epochs,
        ',epochSize'+str(args.epoch_size) if args.epoch_size >0 else '',
        args.batch_size,
        args.lr
    )

    if not args.no_date:
        timestamp=datetime.datetime.now().strftime("%m_%d_%H_%M")
        save_path=os.path.join(timestamp,save_path)
    save_path=os.path.join(args.dataname,save_path)
    print('=> will save everything to {}'.format(save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_writer=SummaryWriter(os.path.join(save_path,'train'))
    test_writer = SummaryWriter(os.path.join(save_path, 'test'))

    input_transform = tramsforms.Compose([
        image_transforms.ArrayToTensor(),
        tramsforms.CenterCrop(180)

    ])

    print("=> fetching img pairs in '{}'".format(args.datadir))

    train_set, test_set = dataset.__dict__[args.dataname](
        args.datadir,
        transform=input_transform,
        split=args.split_file if args.split_file else args.split_value
    )
    print('{} samples found, {} train samples and {} test samples '.format(len(test_set) + len(train_set),
                                                                           len(train_set),
                                                                           len(test_set)))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=False)

    if args.pretrained:
        network_data = torch.load(args.pretrained)
        args.arch = network_data['arch']
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        network_data = None
        print("=> creating model '{}'".format(args.arch))

    mymodel=model.__dict__[args.arch](network_data).cuda()
    mymodel=torch.nn.DataParallel(mymodel).cuda()
    cudnn.benchmark=True

    assert(args.solver in ['adam','sgd'])
    print('=> setting {} solver '.format(args.solver))
    param_groups =[{'params': mymodel.module.bias_parameters(), 'weight_decay': args.bias_decay},
                   {'params': mymodel.module.weight_parameters(),'weight_decay':args.weight_decay}]

    if args.solver == 'adam':
        optimizer=torch.optim.Adam(param_groups,args.lr,betas=(args.momentum,args.beta))
    elif args.solver=='sgd':
        optimizer=torch.optim.SGD(param_groups,args.lr,args.momentum)



if __name__=='__main__':
    main()