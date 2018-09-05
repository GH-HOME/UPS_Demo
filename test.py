
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
import models
import dataset
from matplotlib import pyplot as plt
import datetime
from tensorboardX import SummaryWriter
import numpy as np
import math
from viz_net_pytorch import make_dot

model_names=sorted(name for name in models.__dict__
                   if name.islower() and not name.startswith("__"))


parser = argparse.ArgumentParser(description='PyTorch UPSNet Training on several Datasets',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--datadir', metavar='DIR',default='./dataset/time_09_02_11_55_Light_500_shape_10_albedo_1/UPSDataset',
                    help='path to dataset')
parser.add_argument('--dataname', metavar='DataName',default='Lambertian_direction',
                    help='data set name')
group = parser.add_mutually_exclusive_group()
group.add_argument('-s', '--split-file', default=None, type=str,
                   help='test-val split file')
group.add_argument('--split-value', default=0.8, type=float,
                   help='test-val split proportion (between 0 (only test) and 1 (only train))')

parser.add_argument('--arch', '-a', metavar='ARCH', default='upsnets_bn',
                    choices=model_names,
                    help='model architecture, overwritten if pretrained is specified: ' +
                    ' | '.join(model_names))
parser.add_argument('--solver', default='adam',choices=['adam','sgd'],
                    help='solver algorithms')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epoch-size', default=1000, type=int, metavar='N',
                    help='manual epoch size (will match dataset size if set to 0)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('-sw', '--sparse_weight', default=1, type=float,
                    metavar='W', help='weight for control sparsity in loss')
parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
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
parser.add_argument('--print_intervel',  default=500,
                    help='the iter interval for save the model')
parser.add_argument('--milestones', default=[10,40,80], metavar='N', nargs='*', help='epochs at which learning rate is divided by 2')

best_EPE = -1
n_iter = 0
Light_num=30
ChoiseTime=2000
losstype='angular'


def main():
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

    #args.pretrained='./Lambertian_direction/09_04_19_40/upsnets_bn,adam,300epochs,epochSize1000,b16,lr0.0002/checkpoint.pth.tar'

    if not args.no_date:
        timestamp=datetime.datetime.now().strftime("%m_%d_%H_%M")
        save_path=os.path.join(timestamp,save_path)
    save_path=os.path.join(args.dataname,save_path)
    print('=> will save everything to {}'.format(save_path))
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    train_writer=SummaryWriter(os.path.join(save_path,'train'))
    test_writer = SummaryWriter(os.path.join(save_path, 'test'))

    output_writers = []
    for i in range(3):
        output_writers.append(SummaryWriter(os.path.join(save_path, 'test', str(i))))

    input_transform = image_transforms.Compose([
        image_transforms.ArrayToTensor(),
        image_transforms.CenterCrop(128)

    ])

    print("=> fetching img pairs in '{}'".format(args.datadir))

    train_set, test_set = dataset.__dict__[args.dataname](
        args.datadir,
        transform=input_transform,
        split=args.split_file if args.split_file else args.split_value,
        light_num=Light_num,
        ChoiseTime=ChoiseTime
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

    mymodel=models.__dict__[args.arch](network_data,input_N=Light_num).cuda()
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

    scheduler= torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones,gamma=0.5)

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()

        epoch_adjust_size=12
        if epoch>epoch_adjust_size:
            args.sparse_weight = 0.0
        else:
            args.sparse_weight-=epoch*0.08
        train_loss = train(train_loader, mymodel, optimizer, epoch, train_writer)
        train_writer.add_scalar('mean loss in train epoch', train_loss, epoch)

        eval_loss = validate(val_loader, mymodel, test_writer)
        test_writer.add_scalar('mean loss in test epoch', eval_loss, epoch)

        if eval_loss < 0:
            best_EPE = eval_loss

        is_best = eval_loss < best_EPE
        best_EPE = min(eval_loss, best_EPE)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': mymodel.module.state_dict(),
            'best_EPE': best_EPE
        }, is_best)

        print('=> save model for epoch {}'.format(epoch))


def train(train_loader, mymodel, optimizer, epoch, train_writer):
    global n_iter, args

    batch_time= AverageMeter()
    data_time=AverageMeter()
    losses=AverageMeter()

    epoch_size=len(train_loader) if args.epoch_size==0 else min(len(train_loader),args.epoch_size)

    # switch to train mode
    mymodel.train()

    end=time.time()

    for i, (inputs, target) in enumerate(train_loader):

        #plot the input data
        # plt.title('origin image')
        # plt.imshow(inputs['Imgs'][0][0])
        # plt.show()
        #
        # plt.title('P_Light')
        # plt.imshow(inputs['P_L'][0])
        # plt.show()
        # plt.title('Ground truth light')
        # plt.imshow(target['light'][0])
        # plt.show()

        data_time.update(time.time()-end)
        input_var=inputs
        target_var=target
        for item in inputs.keys():
            input_var[item] = torch.autograd.Variable(inputs[item]).cuda()
        for item in target.keys():
            target_var[item]  = torch.autograd.Variable(target[item]).cuda()
        out_L=mymodel(input_var,train_writer,i,args.print_intervel)



        #plot the output data
        # plt.title('output light image')
        # out_L_show=out_L.detach().cpu().numpy()
        # plt.imshow(out_L_show[0][0])
        # plt.show()

        lossL =calculateLoss_L(out_L,target_var['light'],args.sparse_weight,losstype)
        losses.update(lossL.data[0])
        train_writer.add_scalar('train_loss', lossL.data[0], n_iter)
        n_iter+=1
        optimizer.zero_grad()
        lossL.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_intervel == 0:
            out_L_show=out_L.detach().cpu().numpy()
            out_L_show_=out_L_show[0].reshape(-1,3)
            tar_L_show=target['light'].cpu().numpy()
            tar_L_show_=tar_L_show[0].reshape(-1,3)

            print('Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Loss {5}\t '.format(epoch, i, epoch_size, batch_time, data_time, losses))
            train_writer.add_image('train/Origin image', inputs['Imgs'][0][0],i)
            train_writer.add_image('train/Ground truth light', CreatObservemapFromL(tar_L_show_), i)
            train_writer.add_image('train/Predicted light', CreatObservemapFromL(out_L_show_), i)

        if i >= epoch_size:
            break
    return losses.avg


def calculateLoss_L(input_Lmap,target_Lmap, sparse_weight, type):

    input_Lmap = input_Lmap.view(-1,3).float()

    target_Lmap = target_Lmap.squeeze().float()
    target_Lmap = target_Lmap.view(-1, 3)
    n,_=target_Lmap.shape

    if type=='L2':
        mean_vec = torch.mean(input_Lmap, 0).repeat(n, 1)
        diff_vec = input_Lmap - mean_vec
        distance = torch.sqrt(torch.sum(diff_vec * diff_vec, 1))
        lossfn=torch.nn.MSELoss()
        return lossfn(input_Lmap, target_Lmap)+sparse_weight/distance.mean()
    elif type == 'angular':
        diffangle_cos=torch.abs(1-torch.sum(input_Lmap*target_Lmap.float(),1))
        mean_vec=torch.mean(input_Lmap,0).repeat(n,1)
        diff_vec=input_Lmap-mean_vec
        distance=torch.sqrt(torch.sum(diff_vec * diff_vec,1))

        return (diffangle_cos.mean())+sparse_weight/(distance.mean())
    else:
        raise RuntimeError("no loss type")



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(save_path,filename))
    if is_best:
        shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth.tar'))


def validate(val_loader, mymodel, test_writers):
    global args

    batch_time = AverageMeter()
    loss_eval = AverageMeter()

    # switch to evaluate mode
    mymodel.eval()

    end = time.time()
    with torch.no_grad():
        for i, (inputs, target) in enumerate(val_loader):
            input_var = inputs
            target_var = target
            for item in inputs.keys():
                input_var[item] = torch.autograd.Variable(inputs[item].cuda())
            for item in target.keys():
                target_var[item] = torch.autograd.Variable(target[item].cuda())
            out_L = mymodel(input_var,test_writers,i,args.print_intervel)

            lossL = calculateLoss_L(out_L, target_var['light'], args.sparse_weight,losstype)

            loss_eval.update(lossL)
            test_writers.add_scalar('test_loss', lossL.data[0], i)
            # measure elapsed time

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_intervel == 0:
                out_L_show = out_L.detach().cpu().numpy()
                out_L_show_ = out_L_show[0].reshape(-1, 3)
                tar_L_show = target['light'].cpu().numpy()
                tar_L_show_ = tar_L_show[0].reshape(-1, 3)

                test_writers.add_image('test/Origin image', inputs['Imgs'][0][0], i)
                test_writers.add_image('test/Ground truth light', CreatObservemapFromL(tar_L_show_), i)
                test_writers.add_image('test/Predicted light', CreatObservemapFromL(out_L_show_), i)
                #print('Test: [{0}/{1}]\t Time {2}\t loss {3}'.format(i, len(val_loader), batch_time, loss_eval))


    print(' * loss in evaluation {:.3f}'.format(loss_eval.avg))
    torch.cuda.empty_cache()
    return loss_eval.avg


def CreatObservemapFromL(L=None, scale_size = 32):
    # build observe map for light

    if L is None:
        raise RuntimeError("no light input")
    if type(L) == list:
        L=np.array(L)
    P_observeMap = np.zeros([scale_size, scale_size],np.float)
    for i in range(L.shape[0]):
        x = int((L[i, 0] * 0.5 + 0.5) * (scale_size-2))
        y = int((L[i, 1] * 0.5 + 0.5) * (scale_size-2))
        z = (L[i, 2] * 0.5 + 0.5) * scale_size
        if x>scale_size-1 or x<0 or y>scale_size-1 or y<0:
            continue
        P_observeMap[x, y] = z

    return P_observeMap


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)




if __name__=='__main__':
    main()