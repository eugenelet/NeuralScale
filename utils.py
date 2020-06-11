'''
Some helper functions, e.g. compute parameters, progress bar, etc.
'''
import os
import sys
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

import skimage
import skimage.io
import skimage.transform

import os.path
import hashlib
import errno
from tqdm import tqdm


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None, TOTAL_BAR_LENGTH = 60.):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    try:
        torch.save(state, filename)
        if is_best:
            torch.save(state, filename.replace("checkpoint", "best_model"))
    except:
        print("didn't save checkpoint file")


def adjust_learning_rate(args, optimizer, epoch, search=False, warmup=10):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    if search == True:
        if epoch == warmup:
            args.lr = lr = 0.01
            print("learning rate adjusted: {}".format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if epoch>warmup and (epoch-warmup)%10 == 0:
            lr = args.lr / 1.5**((epoch-warmup)//10)
            print("learning rate adjusted: {}".format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    else:
        if epoch==1:
            lr = args.lr # 0.1
        elif epoch==100:
            lr = args.lr * 0.1**1 # 0.01
        elif epoch==200:
            lr = args.lr * 0.1**2 # 0.001
        elif epoch==250:
            lr = args.lr * 0.1**3 # 0.0001
        if epoch==1 or epoch==100 or epoch==200 or epoch==250:
            print("learning rate adjusted: {}".format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr


def adjust_learning_rate_finetune(args, optimizer, epoch, dataset, search=False, warmup=10):
    """Decay only once at epoch 30 and 15 for CIFAR and tinyimagenet respectively"""
    if dataset!='tinyimagenet':
        if epoch == 20:
            lr = args.lr * 0.1
        elif epoch == 30:
            lr = args.lr * 0.01
        if epoch==20 or epoch==30:
            print("learning rate adjusted: {}".format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    elif dataset=='tinyimagenet':
        if epoch == 10:
            lr = args.lr * 0.1
        elif epoch == 15:
            lr = args.lr * 0.01
        if epoch==10 or epoch==15:
            print("learning rate adjusted: {}".format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr



def adjust_learning_rate_standard(args, optimizer, epoch, dataset, search=False, warmup=10):
    """Decay only once at epoch 30 and 15 for CIFAR and tinyimagenet respectively"""
    if dataset!='tinyimagenet':
        if epoch == 80:
            lr = args.lr * 0.1
        elif epoch == 120:
            lr = args.lr * 0.01
        if epoch==80 or epoch==120:
            print("learning rate adjusted: {}".format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    elif dataset=='tinyimagenet':
        if epoch == 10:
            lr = args.lr * 0.1
        elif epoch == 15:
            lr = args.lr * 0.01
        if epoch==10 or epoch==15:
            print("learning rate adjusted: {}".format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

def adjust_learning_rate_imagenet(args, optimizer, epoch, search=False, warmup=10):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    if search == True:
        if epoch == warmup:
            args.lr = lr = 0.01
            print("learning rate adjusted: {}".format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        if epoch>warmup and epoch%10 == 0:
            lr = args.lr / 1.5**((epoch-warmup)//10)
            print("learning rate adjusted: {}".format(lr))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    else:
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        if epoch%50 == 0:
            lr = args.lr * (0.1 ** (epoch // 50))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print("learning rate adjusted: {}".format(lr))




def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



def compute_params_(model):
    return sum(p.numel() for name, p in model.named_parameters() if p.requires_grad and "gate" not in name)

def compute_params(config, classes=10, model='vgg', in_channel=3, kernel=3, last=True):
    """Computes the number of parameters of CNNs"""
    param_cnt = 0
    if model=='vgg':
        for i, cfg in enumerate(config):
            if i == 0: # first layer
                param_cnt = in_channel * cfg * kernel * kernel + cfg + cfg*2 # weight + bias + BN
                prev_filt = cfg
            else:
                param_cnt += prev_filt * cfg * kernel * kernel + cfg + cfg*2
                prev_filt = cfg
        param_cnt += prev_filt * classes + classes # output/linear layer + bias
    elif model=='resnet18':
        # feed forward
        for i, cfg in enumerate(config):
            if i == 0: # first layer
                param_cnt = in_channel * cfg * kernel * kernel + cfg*2 # weight + BN
                prev_filt = cfg
            else:
                param_cnt += prev_filt * cfg * kernel * kernel + cfg*2
                prev_filt = cfg
        param_cnt += prev_filt * classes + classes # output/linear layer + bias
        # skip connection
        for i in range(2,len(config),2):
            if config[i-2] != config[i]:
                param_cnt += config[i-2] * config[i] # weight
    elif model=='effnetb0':
        filters = [[32],[16],[24,24],[40,40],[80,80,80],[112,112,112],[192,192,192,192],[320],[1280]]
        kernel = [3,3,3,5,3,5,5,3,1]
        expand = [0,1,6,6,6,6,6,6,0]
        last_idx = len(expand) - 1
        filt_cnt = 0
        # feed forward
        for filt,kern,expd in zip(filters,kernel,expand):
            if filt_cnt == 0: # first layer
                param_cnt = in_channel * config[filt_cnt] * kern * kern + config[filt_cnt]*2 # weight + BN
                prev_filt = config[filt_cnt]
                filt_cnt += 1
            elif filt_cnt == last_idx: # output layer
                param_cnt += prev_filt * config[filt_cnt] + config[filt_cnt]*2
                prev_filt = config[filt_cnt]
            else:
                for filt_ in filt:
                    mid_ = prev_filt
                    # expansion
                    if expd != 1:
                        mid_ = prev_filt*expd
                        param_cnt += prev_filt * mid_ + mid_*2
                    # depthwise
                    param_cnt += mid_ * kern*kern + mid_*2
                    # S&E
                    param_cnt += mid_*int(0.25*mid_)*2
                    # conv
                    param_cnt += mid_*config[filt_cnt] + config[filt_cnt]*2
                    prev_filt = config[filt_cnt]
                    filt_cnt += 1
        param_cnt += prev_filt * classes # output/linear layer
    elif model=='mobilenet':
        filters = [32,64,128,128,256,256,512,512,512,512,512,512,1024,1024]
        # feed forward
        for idx, _ in enumerate(filters):
            if idx == 0: # first layer
                param_cnt = in_channel * config[idx] * 3 * 3 + config[idx]*2 # weight + BN
                prev_filt = config[idx]
            else:
                # depthwise
                param_cnt += prev_filt*3*3 + prev_filt*2
                # pointwise
                param_cnt += prev_filt*config[idx] + config[idx]*2
                prev_filt = config[idx]
        param_cnt += prev_filt * classes + classes # linear layer
    elif model=='mobilenetv2':
        if last:
            filters = [[32],[16],[24,24],[32,32,32],[64,64,64,64],[96,96,96],[160,160,160],[320],[1280]]
            expand = [0,1,6,6,6,6,6,6,0]
        else:
            filters = [[32],[16],[24,24],[32,32,32],[64,64,64,64],[96,96,96],[160,160,160],[320]]
            expand = [0,1,6,6,6,6,6,6]
        filt_cnt = 0
        # feed forward
        for filt,expd in zip(filters,expand):
            if filt_cnt == 0: # first layer
                param_cnt = in_channel * config[filt_cnt] + config[filt_cnt] + config[filt_cnt]*2 # weight + bias + BN
                prev_filt = config[filt_cnt]
                filt_cnt += 1
            elif expd==0 and last:
                param_cnt += prev_filt * config[filt_cnt] + config[filt_cnt] + config[filt_cnt]*2
                prev_filt = config[filt_cnt]
            else:
                for filt_ in filt:
                    # shortcut
                    if prev_filt != config[filt_cnt]:
                        param_cnt += prev_filt * config[filt_cnt] + config[filt_cnt] + config[filt_cnt]*2
                    # expansion
                    mid_ = prev_filt*expd
                    param_cnt += prev_filt * mid_ + mid_*2 + mid_
                    # depthwise
                    param_cnt += mid_ * 3*3 + mid_*2 + mid_
                    # conv
                    param_cnt += mid_*config[filt_cnt] + config[filt_cnt] + config[filt_cnt]*2
                    prev_filt = config[filt_cnt]
                    filt_cnt += 1
        if last:
            param_cnt += prev_filt * classes + classes # output/conv layer
        else:
            param_cnt += prev_filt * 1280 + 1280 + 1280*2
            param_cnt += 1280 * classes + classes # output/conv layer
    return param_cnt
