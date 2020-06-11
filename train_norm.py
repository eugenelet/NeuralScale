'''
Training of base network before fine-tuning (for magnitude pruning)
'''

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as utils_data
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import numpy as np
import numpy.linalg as la
import pdb
import pickle
import visdom
import time
import torch.backends.cudnn as cudnn
import gc
import math
import argparse
import copy
from utils import progress_bar, save_checkpoint, adjust_learning_rate, adjust_learning_rate_imagenet, accuracy
import csv
from sklearn import linear_model
from model.VGG import vgg11
from model.preact_resnet import PreActResNet18
from model.resnet import *
from model.lenet import LeNet
from model.mobilenetv2 import MobileNetV2
from model.mobilenet import MobileNet
from torch.optim.lr_scheduler import StepLR
import os
from copy import deepcopy


##############
## Function ##
##############
def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features



def train(args, model, train_loader, optimizer, epoch, criterion, pruning_engine=None, scheduler=None):
    """Train for one epoch on the training set also performs pruning"""
    train_loss = 0
    train_acc = 0

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        # make sure that all gradients are zero
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

        output = model(data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))


        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += prec1.item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%%'
            % (train_loss/(batch_idx+1), train_acc/(batch_idx+1)))

    return train_acc/(batch_idx+1), train_loss/(batch_idx+1)


def validate(args, test_loader, model, criterion, epoch, pruning_engine=None, optimizer=None):
    """Perform validation on the validation set"""
    test_loss = 0
    test_acc = 0
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data = data.cuda()
            target = target.cuda()
            output = model(data)

            loss = criterion(output, target)

            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            
            test_loss += loss.item()
            test_acc += prec1.item()
            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%%'
                % (test_loss/(batch_idx+1), test_acc/(batch_idx+1)))

    return test_acc/(batch_idx+1), test_loss/(batch_idx+1)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Train a base network under ratio=1 (default configuration) for pruning')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 40)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    # parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
    #                     help='learning rate (default: 0.01)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--lr-decay-every', type=int, default=100,
                        help='learning rate decay by 10 every X epochs')
    parser.add_argument('--lr-decay-scalar', type=float, default=0.1,
                        help='--')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--dataset', default="CIFAR10", type=str,
                            help='dataset for experiment, choice: MNIST, CIFAR10', choices= ["MNIST", "CIFAR10", "CIFAR100", "Imagenet", "tinyimagenet"])
    parser.add_argument('--data', metavar='DIR', default='/DATA/tiny-imagenet-200', help='path to tinyimagenet dataset')
    parser.add_argument('--model', default="resnet18", type=str,
                        help='model selection, choices: vgg, mobilenetv2, resnet18',
                        choices=["vgg", "mobilenetv2", "resnet18", "mobilenet"])
    parser.add_argument('--r', dest="resume", action='store_true', default=False,
                        help='Resume from checkpoint')
    parser.add_argument('--save', default='model',
                        help='model file')
    parser.add_argument('--prune_fname', default='filename',
                        help='prune save file')
    parser.add_argument('--descent_idx', type=int, default=14,
                        help='Iteration for Architecture Descent')
    parser.add_argument('--s', type=float, default=0.0001,
                        help='scale sparse rate (default: 0.0001)')

    args = parser.parse_args()

    ##################
    ## Data loading ##
    ##################

    kwargs = {'num_workers': 1, 'pin_memory': True}
    if(args.dataset == "CIFAR10"):
        print("Using Cifar10 Dataset")
        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                            std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        trainset = torchvision.datasets.CIFAR10(root='/DATA/data_cifar10/', train=True, 
                                                download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                    shuffle=True, **kwargs)
        testset = torchvision.datasets.CIFAR10(root='/DATA/data_cifar10/', train=False,
                                                download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                    shuffle=True, **kwargs)
    elif args.dataset == "CIFAR100":
        print("Using Cifar100 Dataset")
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        trainset = torchvision.datasets.CIFAR100(root='/DATA/data_cifar100/', train=True, 
                                                download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                    shuffle=True, **kwargs)
        testset = torchvision.datasets.CIFAR100(root='/DATA/data_cifar100/', train=False,
                                                download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                                    shuffle=True, **kwargs)
    elif args.dataset == "Imagenet":
        print("Using Imagenet Dataset")
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

        train_dataset = torchvision.datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

        train_sampler = None

        kwargs = {'num_workers': 16}

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            sampler=train_sampler, pin_memory=True, **kwargs)

        if args.use_test_as_train:
            train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=(train_sampler is None), **kwargs)


        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False, pin_memory=True, **kwargs)
    elif args.dataset == "tinyimagenet":
        print("Using tiny-Imagenet Dataset")
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'test')

        normalize = transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        train_dataset = torchvision.datasets.ImageFolder(
                            traindir,
                            transforms.Compose([
                                transforms.RandomCrop(64, padding=4),
                                transforms.RandomRotation(20),
                                # transforms.RandomResizedCrop(64),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize,
                            ]))

        train_sampler = None

        kwargs = {'num_workers': 16}

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            sampler=train_sampler, pin_memory=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(valdir, transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False, pin_memory=True, **kwargs)
    else:
        print("Dataset does not exist! [CIFAR10, MNIST, tinyimagenet]")
        exit()

    if args.dataset=='CIFAR10':
        num_classes = 10
    elif args.dataset=='CIFAR100':
        num_classes = 100
    elif args.dataset=='tinyimagenet':
        num_classes = 200

    ratio = 1
    pruned_filters = None
    ###########
    ## Model ##
    ###########
    print("Setting Up Model...")
    if args.model == "vgg":
        model = vgg11(ratio=ratio, neuralscale=False, num_classes=num_classes, prune_fname=args.prune_fname, descent_idx=args.descent_idx, pruned_filters=pruned_filters)
    elif args.model == "resnet18":
        model = PreActResNet18(ratio=ratio, neuralscale=False, num_classes=num_classes, dataset=args.dataset, prune_fname=args.prune_fname, descent_idx=args.descent_idx, pruned_filters=pruned_filters, search=True)
    elif args.model == "mobilenetv2":
        model = MobileNetV2(ratio=ratio, neuralscale=False, num_classes=num_classes, dataset=args.dataset, prune_fname=args.prune_fname, descent_idx=args.descent_idx, pruned_filters=pruned_filters, search=True)
    else:
        print(args.model, "model not supported")
        exit()
    print("{} set up.".format(args.model))


    # for model saving
    model_path = "saved_models"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    log_save_folder = "%s/%s"%(model_path, args.model)
    if not os.path.exists(log_save_folder):
        os.makedirs(log_save_folder)

    model_save_path = "%s/%s"%(log_save_folder, args.save) + "_checkpoint.t7"
    model_state_dict = model.state_dict()
    if args.save:
        print("Model will be saved to {}".format(model_save_path))
        save_checkpoint({
            'state_dict': model_state_dict
        }, False, filename = model_save_path)
    else:
        print("Save path not defined. Model will not be saved.")

    # Assume cuda is available and uses single GPU
    model.cuda()
    cudnn.benchmark = True

    # define objective
    criterion = nn.CrossEntropyLoss()

    
    ######################
    ## Set up pruning   ##
    ######################
    # remove updates from gate layers, because we want them to be 0 or 1 constantly
    parameters_for_update = []
    parameters_for_update_named = []
    for name, m in model.named_parameters():
        if "gate" not in name:
            parameters_for_update.append(m)
            parameters_for_update_named.append((name, m))
        else:
            print("skipping parameter", name, "shape:", m.shape)

    total_size_params = sum([np.prod(par.shape) for par in parameters_for_update])
    print("Total number of parameters, w/o usage of bn consts: ", total_size_params)

    optimizer = optim.SGD(parameters_for_update, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    ###############
    ## Training  ##
    ###############
    best_test_acc = 0
    train_acc_plt = []
    train_loss_plt = []
    test_acc_plt = []
    test_loss_plt = []
    epoch_plt = []
    if num_classes == 200: # tinyimagenet
        args.epochs = 150
    else:
        args.epochs = 300
    for epoch in range(1, args.epochs + 1):
        if num_classes == 200: # tinyimagenet
            adjust_learning_rate_imagenet(args, optimizer, epoch, search=False)
        else:
            adjust_learning_rate(args, optimizer, epoch)

        print("Epoch: {}".format(epoch))

        # train model
        train_acc, train_loss = train(args, model, train_loader, optimizer, epoch, criterion)

        # evaluate on validation set
        test_acc, test_loss = validate(args, test_loader, model, criterion, epoch, optimizer=optimizer)

        # remember best prec@1 and save checkpoint
        is_best = test_acc > best_test_acc
        best_test_acc = max(test_acc, best_test_acc)
        model_state_dict = model.state_dict()
        if args.save:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_state_dict,
                'best_prec1': test_acc,
            }, is_best, filename=model_save_path)


        train_acc_plt.append(train_acc)
        train_loss_plt.append(train_loss)
        test_acc_plt.append(test_acc)
        test_loss_plt.append(test_loss)
        epoch_plt.append(epoch)


    pickle_save = {
        "ratio": ratio,
        "train_acc": train_acc_plt,
        "train_loss": train_loss_plt,
        "test_acc": test_acc_plt,
        "test_loss": test_loss_plt,
    }
    plot_path = "saved_plots"
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    log_save_folder = "%s/%s"%(plot_path, args.model)
    if not os.path.exists(log_save_folder):
        os.makedirs(log_save_folder)

    pickle_out = open("%s/%s_%s.pk"%(log_save_folder, args.save, int(ratio*100)),"wb")
    pickle.dump(pickle_save, pickle_out)
    pickle_out.close()

if __name__ == '__main__':
    main()
