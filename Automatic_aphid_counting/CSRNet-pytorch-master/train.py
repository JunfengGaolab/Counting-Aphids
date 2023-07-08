import sys
import os

import warnings

from model import CSRNet

from utils import save_checkpoint

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
import argparse
import json
import cv2
import dataset
import time
from visdom import Visdom
from math import sqrt

parser = argparse.ArgumentParser(description='PyTorch CSRNet')

parser.add_argument('train_json', metavar='TRAIN',
                    help='path to train json')

parser.add_argument('val_json', metavar='VAL',
                    help='path to val json')
#parser.add_argument('test_json', metavar='TEST',
                    #help='path to test json')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')

parser.add_argument('gpu',metavar='GPU', type=str,
                    help='GPU id to use.')

parser.add_argument('task',metavar='TASK', type=str,
                    help='task id to use.')

def main():
    
    global args,best_prec1
    
    best_prec1 = 1e6
    
    args = parser.parse_args()
    #args.original_lr = 1e-7 #0.01
    #args.lr = 1e-7
    args.batch_size    = 1 #if the size of images are not equel, we have to set as 1
    args.batch_size_test    = 1
    args.momentum      = 0.95 # 0.937
    args.decay         = 5*1e-4 # weight_decay: 0.0005
    args.start_epoch   = 0

    #args.original_lr = 1e-7  # 0.01
    #args.lr = 1e-7
    #args.epochs = 400 #400
    #args.steps         = [-1,1,100,150]
    #args.scales        = [1,1,1,1]

    args.original_lr = 1e-5
    args.lr = 1e-5
    args.epochs = 2000
    args.steps = [500, 1200, 1600]
    args.scales = [0.1, 0.1, 0.1]

    args.workers = 4
    args.seed = time.time()
    args.print_freq = 30
    with open(args.train_json, 'r',encoding='UTF-8-sig') as outfile:
        #if outfile.startswith(u'\ufeff'):
            #outfile = outfile.encode('utf8')[3:].decode('utf8')        
        train_list = json.load(outfile)
        print('train_list:',train_list)


    with open(args.val_json, 'r',encoding='UTF-8-sig') as outfile:
        val_list = json.load(outfile)
        print('val_list:',val_list)

    '''
    with open(args.test_json, 'r',encoding='UTF-8-sig') as outfile:
        test_list = json.load(outfile)
        print('test_list:',test_list)
    '''
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)
    
    model = CSRNet()

    model = model.cuda()
    
    criterion = nn.MSELoss(size_average=False).cuda()
    
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)

    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    # 将窗口类实例化
    viz = Visdom()
    # 创建窗口并初始化
    viz.line([0.], [0], win='train_loss', opts=dict(title='train_loss'))
    viz.line([0.], [0], win='val_loss', opts=dict(title='val_loss'))
    viz.line([0.], [0], win='val_MAE', opts=dict(title='val_MAE'))
    viz.line([0.], [0], win='val_MSE', opts=dict(title='val_MSE'))
    viz.line([0.], [0], win='val_RMSE', opts=dict(title='val_RMSE'))

    txtName1 = "./result/train_loss.txt"
    training_loss = open(txtName1, "w")

    txtName2 = "./result/val_loss.txt"
    validation_loss = open(txtName2, "w")

    txtName3 = "./result/val_mae.txt"
    val_mae = open(txtName3, "w")

    txtName4 = "./result/val_mse.txt"
    val_mse = open(txtName4, "w")

    txtName5 = "./result/val_rmse.txt"
    val_rmse = open(txtName5, "w")


    for epoch in range(args.start_epoch, args.epochs):
        
        adjust_learning_rate(optimizer, epoch)
        
        train_loss = train(train_list, model, criterion, optimizer, epoch)
        #val_loss,prec1 = validate(val_list, model, criterion)

        val_loss,prec1,mse,rmse = validate(val_list, model, criterion)



        str_train_loss=str(float(train_loss))
        str_val_loss = str(float(val_loss))
        str_val_mae = str(float(prec1))
        str_val_mse = str(float(mse))
        str_val_rmse = str(float(rmse))
        #print('str_train_loss',str_train_loss)
        #print('str_val_loss', str_val_loss)
        #print('str_val_mae', str_val_mae)
        #print('str_val_mse', str_val_mse)
        #print('str_val_rmse', str_val_rmse)

        training_loss.write(str_train_loss + '\n')
        validation_loss.write(str_val_loss + '\n')
        val_mae.write(str_val_mae + '\n')
        val_mse.write(str_val_mse + '\n')
        val_rmse.write(str_val_rmse + '\n')

        viz.line([float(train_loss)], [epoch], win='train_loss', update='append')
        viz.line([float(val_loss)], [epoch], win='val_loss', update='append')
        viz.line([float(prec1)], [epoch], win='val_MAE', update='append')
        viz.line([float(mse)], [epoch], win='val_MSE', update='append')
        viz.line([float(rmse)], [epoch], win='val_RMSE', update='append')



        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f}'
              .format(mae=best_prec1))
        print('epoch:',epoch)
        #best_model
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best,args.task)


        torch.save(model.state_dict(), os.path.join("./models", "model_final.pth"))#final_model


        if epoch%200==0 and epoch!=0:
            torch.save(model.state_dict(),os.path.join("./models",str(epoch)+ "_model.pth"))

    training_loss.close()
    validation_loss.close()
    val_mae.close()
    val_mse.close()
    val_rmse.close()

def train(train_list, model, criterion, optimizer, epoch):
    
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                       shuffle=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]), 
                       train=True, 
                       seen=model.seen,
                       batch_size=args.batch_size,
                       num_workers=args.workers),
        batch_size=args.batch_size)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))
    
    model.train()
    end = time.time()

    train_epoch_loss=0

    for i,(img, target)in enumerate(train_loader):
        data_time.update(time.time() - end)
        #print('i',i)
        img = img.cuda()
        img = Variable(img,volatile=False)

        target = target.type(torch.FloatTensor).unsqueeze(0).cuda()
        target = Variable(target, volatile=False)

        optimizer.zero_grad()

        output = model(img)

        loss = criterion(output, target)
        train_epoch_loss += loss.item()
        losses.update(loss.item(), img.size(0))
        ##optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        batch_time.update(time.time() - end)
        end = time.time()


        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))

    return (train_epoch_loss/len(train_loader))
    
def validate(val_list, model, criterion):
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
    batch_size=args.batch_size_test)    
    
    model.eval()
    
    mae = 0
    mse = 0
    rmse = 0
    test_epoch_loss = 0
    with torch.no_grad():
        for i,(img, target) in enumerate(test_loader):
            img = img.cuda()
            img = Variable(img,volatile=False)
            output = model(img)

            target = target.type(torch.FloatTensor).unsqueeze(0).cuda()
            target = Variable(target,volatile=False)

            loss = criterion(output, target)
            test_epoch_loss += loss.item()


            mae += abs(output.data.sum()-target.sum().type(torch.FloatTensor).cuda())

            mse += (target.sum().type(torch.FloatTensor).cuda() - output.data.sum()) * (
                        target.sum().type(torch.FloatTensor).cuda() - output.data.sum())

    mae = mae/len(test_loader)
    mse = mse / len(test_loader)
    rmse = sqrt(mse / len(test_loader))
    print(' * MAE {mae:.3f} '
              .format(mae=mae))

    return (test_epoch_loss/len(test_loader)),mae,mse,rmse
        
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    
    
    args.lr = args.original_lr
    
    for i in range(len(args.steps)):
        
        scale = args.scales[i] if i < len(args.scales) else 1
        
        
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
        
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
    
if __name__ == '__main__':
    main()        
