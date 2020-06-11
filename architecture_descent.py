'''
Peform architecure descent on VGG11, ResNet18 and MobileNetV2 using CIFAR10/100.
Runs for 15 iterations.
'''

from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pickle
import torch.backends.cudnn as cudnn
import argparse
from utils import progress_bar, save_checkpoint, adjust_learning_rate, accuracy
from model.VGG import vgg11
from model.preact_resnet import PreActResNet18
from model.mobilenetv2 import MobileNetV2
from prune import pruner, prepare_pruning_list

import os
from fit_params import fit_params

##############
## Function ##
##############
def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features



def train(args, model, train_loader, optimizer, epoch, criterion, pruning_engine=None, num_classes=100):
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

        loss.backward(retain_graph=True)


        optimizer.step()

        train_loss += loss.item()
        train_acc += prec1.item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%%'
            % (train_loss/(batch_idx+1), train_acc/(batch_idx+1)))

        if args.prune_on and epoch>args.warmup:
            ret_val = pruning_engine.do_step(loss=loss.item(), optimizer=optimizer)
            if (ret_val == -1):
                if pruning_engine.recorded_filters == []: # not all layers pruned at least once, opt for uniform scaling
                    pruning_engine.report_filter_number(force_save=True)
                pruning_engine.writer.writerows(pruning_engine.recorded_filters)
                pruning_engine.f.close()
                return -1, -1 # terminate training
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
            if args.prune_on:
                pruning_engine.do_step(loss=loss.item(), optimizer=optimizer)

    return test_acc/(batch_idx+1), test_loss/(batch_idx+1)



def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Search for optimal configuration using architecture descent')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 40)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--dataset', default="CIFAR10", type=str,
                            help='dataset for experiment, choice: CIFAR10, CIFAR100', choices= ["CIFAR10", "CIFAR100"])
    parser.add_argument('--model', default="vgg", type=str,
                        help='model selection, choices: vgg, mobilenetv2, resnet18',
                        choices=["vgg", "mobilenetv2", "resnet18"])
    parser.add_argument('--save', default='model',
                        help='model and prune file')
    parser.add_argument('--no_prune', dest="prune_on", action='store_false', default=True,
                        help='Turn off pruning')
    parser.add_argument('--warmup', type=int, default=10,
                        help='number of warm-up or fine-tuning epochs before pruning (default: 10)')
    parser.add_argument('--morph', dest="morph", action='store_true', default=False,
                        help='Prunes only 50 percent of neurons, for comparison with MorphNet')
    args = parser.parse_args()

    ##################
    ## Data loading ##
    ##################
    kwargs = {'num_workers': 1, 'pin_memory': True}
    if(args.dataset == "CIFAR10"):
        print("Using Cifar10 Dataset")
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

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False, pin_memory=True, **kwargs)
    else:
        print("Dataset does not exist! [CIFAR10, CIFAR100]")
        exit()

    if args.dataset == "CIFAR10":
        num_classes = 10
    elif args.dataset == "CIFAR100":
        num_classes = 100

    param_list = []
    if args.morph:
        total_iter = 1
    else:
        total_iter = 15
    for iteration in range(total_iter):
        print("Iteration: {}".format(iteration))
        args.lr = 0.1
        ###########
        ## Model ##
        ###########
        print("Setting Up Model...")
        if args.model == "vgg":
            model = vgg11(ratio=1, neuralscale=True, iteration=iteration, num_classes=num_classes, search=True)
        elif args.model == "resnet18":
            model = PreActResNet18(ratio=1, neuralscale=True, iteration=iteration, num_classes=num_classes, search=True, dataset=args.dataset)
        elif args.model == "mobilenetv2":
            model = MobileNetV2(ratio=1, neuralscale=True, iteration=iteration, num_classes=num_classes, search=True, dataset=args.dataset)
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


        if args.prune_on:
            pruning_parameters_list = prepare_pruning_list(model)
            print("Total pruning layers:", len(pruning_parameters_list))
            if args.morph:
                prune_neurons = 0.5
            else:
                prune_neurons = 0.95
            pruning_engine = pruner(pruning_parameters_list, iteration=iteration, prune_fname=args.save, classes=num_classes, model=args.model, prune_neurons=prune_neurons)


        ###############
        ## Training  ##
        ###############
        for epoch in range(1, args.epochs + 1):
            print("Epoch: {}".format(epoch))
            adjust_learning_rate(args, optimizer, epoch, search=True, warmup=args.warmup)
            # train model
            if args.prune_on:
                train_acc, train_loss = train(args, model, train_loader, optimizer, epoch, criterion, pruning_engine=pruning_engine, num_classes=num_classes)
                if train_acc==-1 and train_loss==-1:
                    break
            else:
                train_acc, train_loss = train(args, model, train_loader, optimizer, epoch, criterion, num_classes=num_classes)
        
        # =========================
        # store searched parameters
        # =========================
        alpha, beta = fit_params(iteration=iteration, prune_fname=args.save, classes=num_classes, model=args.model) # search for params of each layer
        param_list.append([alpha,beta])
    
    # =======================
    # save scaling parameters
    # =======================
    pickle_save = {
        "param": param_list,
    }

    pickle_out = open("prune_record/" + args.save + ".pk","wb")
    pickle.dump(pickle_save, pickle_out)
    pickle_out.close()

if __name__ == '__main__':
    main()
