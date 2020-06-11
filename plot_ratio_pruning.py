'''
Plots accuracy obtained using train-prune-finetune pipeline
'''



from matplotlib import pyplot as plt
import numpy as np
import pickle
import argparse
from utils import compute_params_
from model.VGG import vgg11, VGG
from model.preact_resnet import PreActResNet18, PreActResNet
from model.mobilenetv2 import MobileNetV2
from prune import pruner, prepare_pruning_list#, setup_latency
import torchvision.transforms as transforms
import torchvision
import torch
from thop import profile
import time
import os

parser = argparse.ArgumentParser(description='Plots accuracy obtained using train-prune-finetune pipeline')
parser.add_argument('--dataset', default="CIFAR100", type=str,
                        help='dataset for experiment, choice: CIFAR10, CIFAR100, tinyimagenet', choices= ["CIFAR10", "CIFAR100", "tinyimagenet"])
parser.add_argument('--model', default="vgg", type=str,
                    help='model selection, choices: vgg, mobilenetv2, resnet18',
                    choices=["lenet", "vgg", "mobilenetv2", "resnet18"])
parser.add_argument('--latency_on', dest="latency_on", action='store_true', default=False,
                    help='Use latency regularizer for pruning')

args = parser.parse_args()

if args.dataset=="CIFAR100":
    num_classes = 100
elif args.dataset=="tinyimagenet":
    num_classes = 200
elif args.dataset=="CIFAR10":
    num_classes = 10

if(args.dataset == "CIFAR10"):
    print("Using Cifar10 Dataset")
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                        std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    testset = torchvision.datasets.CIFAR10(root='/DATA/data_cifar10/', train=False,
                                            download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100,
                                                shuffle=True)
elif args.dataset == "CIFAR100":
    print("Using Cifar100 Dataset")
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    testset = torchvision.datasets.CIFAR100(root='/DATA/data_cifar100/', train=False,
                                            download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=100,
                                                shuffle=False)
elif args.dataset == "tinyimagenet":
    print("Using TinyImageNet Dataset")
    valdir = os.path.join("/DATA/tiny-imagenet-200", 'test')

    normalize = transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
    kwargs = {'num_workers': 16}

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(valdir, transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=100, shuffle=False, pin_memory=True)
else:
    print("Dataset does not exist! [Imagenet]")
    exit()


data, target = next(test_loader.__iter__())
data = data.cuda()
def compute_latency(model):
    latency = list()
    model = model.cuda()
    model.eval()
    last_time = time.time()
    for idx in range(100):
        with torch.no_grad():
            _ = model(data)
        cur_time = time.time()
        if idx > 20: # allow 20 runs for GPU to warm-up
            latency.append(cur_time - last_time)
        last_time = cur_time
    del model
    return np.mean(latency) * 1000

if args.model=="vgg":
    config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    new_config = VGG.prepare_filters(VGG, config, ratio=0.75, neuralscale=False, num_classes=num_classes)
    model = vgg11(config=new_config, num_classes=num_classes)
    latency = compute_latency(model)
    params = compute_params_(model)
elif args.model == "resnet18":
    filters = [[64],[64,64],[64,64],[128,128],[128,128],[256,256],[256,256],[512,512],[512,512]]
    new_config = PreActResNet.prepare_filters(PreActResNet, filters, ratio=0.75, neuralscale=False, num_classes=num_classes)
    model = PreActResNet18(filters=new_config, num_classes=num_classes, dataset=args.dataset)
    latency = compute_latency(model)
    params = compute_params_(model)
elif args.model == "mobilenetv2":
    filters = [[32],[16],[24,24],[32,32,32],[64,64,64,64],[96,96,96],[160,160,160],[320],[1280]]
    new_config = MobileNetV2.prepare_filters(MobileNetV2, filters, ratio=0.75, neuralscale=False, num_classes=num_classes)
    model = MobileNetV2(filters=new_config, num_classes=num_classes, dataset=args.dataset)
    latency = compute_latency(model)
    params = compute_params_(model)
        
ratio = 0.75

uni_test_loss = []
uni_test_acc = []

uni_test_loss_tmp = []
uni_test_acc_tmp = []
num_samples = 5
for i in range(num_samples):
    if args.model == "vgg":
        if args.dataset == "CIFAR100":
            pkl_ld = pickle.load( open( "saved_plots/vgg/vgg_norm_c100_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
            uni_test_loss_tmp.append(min(pkl_ld["test_loss"]))
            uni_test_acc_tmp.append(max(pkl_ld["test_acc"]))
        elif args.dataset == "CIFAR10":
            # Pruned (Uniform Scale)
            pkl_ld = pickle.load( open( "saved_plots/vgg/vgg_norm_c10_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
            uni_test_loss_tmp.append(min(pkl_ld["test_loss"]))
            uni_test_acc_tmp.append(max(pkl_ld["test_acc"]))
        else:
            print("Dataset Not Found...")
            exit()
    elif args.model == "resnet18":
        if args.dataset == "CIFAR100":
            pkl_ld = pickle.load( open( "saved_plots/resnet18/resnet18_norm_c100_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
            uni_test_loss_tmp.append(min(pkl_ld["test_loss"]))
            uni_test_acc_tmp.append(max(pkl_ld["test_acc"]))
        elif args.dataset == "tinyimagenet":
            pkl_ld = pickle.load( open( "saved_plots/resnet18/resnet18_norm_tinyimagenet_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
            uni_test_loss_tmp.append(min(pkl_ld["test_loss"]))
            uni_test_acc_tmp.append(max(pkl_ld["test_acc"]))
        else:
            print("Dataset Not Found...")
    elif args.model == "mobilenetv2":
        if args.dataset == "CIFAR100":
            pkl_ld = pickle.load( open( "saved_plots/mobilenetv2/mobilenetv2_norm_c100_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
            uni_test_loss_tmp.append(min(pkl_ld["test_loss"]))
            uni_test_acc_tmp.append(max(pkl_ld["test_acc"]))
        elif args.dataset == "tinyimagenet":
            pkl_ld = pickle.load( open( "saved_plots/mobilenetv2/mobilenetv2_norm_tinyimagenet_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
            uni_test_loss_tmp.append(min(pkl_ld["test_loss"]))
            uni_test_acc_tmp.append(max(pkl_ld["test_acc"]))
        else:
            print("Dataset Not Found...")


uni_test_loss_tmp = np.array(uni_test_loss_tmp)
uni_test_acc_tmp = np.array(uni_test_acc_tmp)

uni_test_loss_max = uni_test_loss_tmp.max(axis=0)
uni_test_acc_max = uni_test_acc_tmp.max(axis=0)

uni_test_loss_min = uni_test_loss_tmp.min(axis=0)
uni_test_acc_min = uni_test_acc_tmp.min(axis=0)

uni_test_loss_std = uni_test_loss_tmp.std(axis=0)
uni_test_acc_std = uni_test_acc_tmp.std(axis=0)

uni_test_loss = uni_test_loss_tmp.mean(axis=0)
uni_test_acc = uni_test_acc_tmp.mean(axis=0)
    


print("===============")
print("Comparison Table")
print("===============")
print("Param    latency   Accuracy    std")
print("Pruned")
print("{}      {}    {}      {}".format(params, latency, uni_test_acc, uni_test_acc_std))
