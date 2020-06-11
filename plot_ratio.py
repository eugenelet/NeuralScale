'''
Plots accuracy of different methods under various parameter scale.
'''
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pickle
import argparse
from utils import compute_params_
from model.VGG import vgg11, VGG
from model.preact_resnet import PreActResNet18, PreActResNet
from model.mobilenetv2 import MobileNetV2
from prune import pruner, prepare_pruning_list
import torchvision.transforms as transforms
import torchvision
import torch
import time
import os

parser = argparse.ArgumentParser(description='Plots accuracy of different methods under various parameter scale')
parser.add_argument('--dataset', default="CIFAR100", type=str,
                        help='dataset for experiment, choice: CIFAR10, CIFAR100, tinyimagenet', choices= ["CIFAR10", "CIFAR100", "tinyimagenet"])
parser.add_argument('--model', default="vgg", type=str,
                    help='model selection, choices: vgg, mobilenetv2, resnet18',
                    choices=["vgg", "mobilenetv2", "resnet18"])
parser.add_argument('--convcut', dest="convcut", action='store_true', default=False,
                    help='Show comparison with network that use convolutional layer for all shortcut layer (only for ResNet18 and MobileNetV2)')
parser.add_argument('--plot_normal', dest="plot_normal", action='store_true', default=False,
                    help='Plot using normal size without considering size constraints for publication')

args = parser.parse_args()


width = 496.85625 # textwidth of CVPR2020
colwidth = 237.13594 # columnwidth CVPR2020

def set_size(width, fraction=1):
    """ Set aesthetic figure dimensions to avoid scaling in latex.

    Parameters
    ----------
    width: float
            Width in pts
    fraction: float
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    if args.convcut or args.plot_normal:
        golden_ratio = (5**.5 - 0.5) / 2
    else:
        golden_ratio = (5**.5 - 0.85) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


# Settings for plot fonts
if args.convcut or args.plot_normal:
    nice_fonts = {
            # Use LaTeX to write all text
            "text.usetex": True,
            # "font.family": "Times New Roman",
            # Use 10pt font in plots, to match 10pt font in document
            "axes.linewidth": 10 / 12.,
            "lines.linewidth": 10 / 12.,
            "lines.markersize": 30 / 12.,
            "patch.linewidth": 10 / 12.,
            "axes.labelsize": 10,
            "font.size": 10,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
    }
else:
    nice_fonts = {
            # Use LaTeX to write all text
            "text.usetex": True,
            # "font.family": "Times New Roman",
            # Use 10pt font in plots, to match 10pt font in document
            "axes.linewidth": 4 / 12.,
            "lines.linewidth": 4 / 12.,
            "lines.markersize": 12 / 12.,
            "patch.linewidth": 4 / 12.,
            "axes.labelsize": 4,
            "font.size": 4,
            # Make the legend/label fonts a little smaller
            "legend.fontsize": 3.5,
            "xtick.labelsize": 3,
            "ytick.labelsize": 3,
    }

mpl.rcParams.update(nice_fonts)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


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
    print("Using tiny-Imagenet Dataset")
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


def compute_latency(model):
    latency = list()
    model = model.cuda()
    model.eval()
    last_time = time.time()
    data, target = next(test_loader.__iter__())
    data = data.cuda()
    for idx in range(100):
        with torch.no_grad():
            _ = model(data)
        cur_time = time.time()
        if idx > 20: # allow 20 runs for GPU to warm-up
            latency.append(cur_time - last_time)
        last_time = cur_time
    del model
    del data
    torch.cuda.empty_cache()
    return np.mean(latency) * 1000

param_low = []
param_high = []
param_convcut = []
param_uni = []
param_prune = []

latency_low = []
latency_high = []
latency_convcut = []
latency_uni = []
latency_prune = []

if args.model=="vgg":
    config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    if args.dataset == "CIFAR10":
        config_prune = [64, 128, 249, 253, 268, 175, 87, 152] # VGG C10
    elif args.dataset == "CIFAR100":
        config_prune = [63, 125, 204, 215, 234, 174, 120, 241] # VGG C100
    ratios = np.arange(0.25,2.1,0.25) # [0.25, 0.5 , 0.75, 1, 1.25, 1.5 , 1.75, 2]
    for ratio in ratios:
        # uniform
        new_config = VGG.prepare_filters(VGG, config, ratio=ratio, neuralscale=False, num_classes=num_classes)
        model = vgg11(config=new_config, num_classes=num_classes)
        latency = compute_latency(model)
        params = compute_params_(model)
        param_uni.append(params)
        latency_uni.append(latency)
        # pruned
        new_config = VGG.prepare_filters(VGG, config, ratio=ratio, neuralscale=False, num_classes=num_classes, pruned_filters=config_prune)
        model = vgg11(config=new_config, num_classes=num_classes)
        latency = compute_latency(model)
        params = compute_params_(model)
        param_prune.append(params)
        latency_prune.append(latency)
        ## efficient
        if args.dataset == "CIFAR100":
            fname = "vgg2_10_eff_c100"
            # fname = "vgg_eff_c100"
            fname_convcut = "vgg100_c100_01"
        elif args.dataset == "CIFAR10":
            fname = "vgg_10_eff_c10"
            fname_convcut = "vgg_c10_01"
        # low
        new_config = VGG.prepare_filters(VGG, config, ratio=ratio, neuralscale=True, descent_idx=0, prune_fname=fname, num_classes=num_classes)
        model = vgg11(ratio=ratio, neuralscale=True, num_classes=num_classes, prune_fname=fname, descent_idx=0)
        latency = compute_latency(model)
        params = compute_params_(model)
        param_low.append(params)
        latency_low.append(latency)
        # high
        new_config = VGG.prepare_filters(VGG, config, ratio=ratio, neuralscale=True, descent_idx=14, prune_fname=fname, num_classes=num_classes)
        model = vgg11(ratio=ratio, neuralscale=True, num_classes=num_classes, prune_fname=fname, descent_idx=14)
        latency = compute_latency(model)
        params = compute_params_(model)
        param_high.append(params)
        latency_high.append(latency)
elif args.model == "resnet18":
    filters = [[64],[64,64],[64,64],[128,128],[128,128],[256,256],[256,256],[512,512],[512,512]]
    if args.dataset == "CIFAR100":
        filters_prune = [48, 46, 40, 41, 54, 91, 75, 73, 95, 157, 149, 149, 156, 232, 216, 140, 190] # resnet18 c100
        ratios = np.arange(0.25,2.1,0.25) # [0.25, 0.5 , 0.75, 1, 1.25, 1.5 , 1.75, 2]
    elif args.dataset == "tinyimagenet":
        filters_prune = [82,90,78,80,96,180,104,96,194,312,182,178,376,546,562,454,294]
        ratios = [0.25, 0.5, 0.75, 1.0]
    for ratio in ratios:
        # convcut
        new_config = PreActResNet.prepare_filters(PreActResNet, filters, ratio=ratio, neuralscale=False, num_classes=num_classes)
        model = PreActResNet18(filters=new_config, num_classes=num_classes, dataset=args.dataset, convcut=True)
        latency = compute_latency(model)
        params = compute_params_(model)
        param_convcut.append(params)
        latency_convcut.append(latency)
        # uniform
        new_config = PreActResNet.prepare_filters(PreActResNet, filters, ratio=ratio, neuralscale=False, num_classes=num_classes)
        model = PreActResNet18(filters=new_config, num_classes=num_classes, dataset=args.dataset)
        latency = compute_latency(model)
        params = compute_params_(model)
        param_uni.append(params)
        latency_uni.append(latency)
        # pruned
        new_config = PreActResNet.prepare_filters(PreActResNet, filters, ratio=ratio, neuralscale=False, num_classes=num_classes, pruned_filters=filters_prune)
        model = PreActResNet18(filters=new_config, num_classes=num_classes, dataset=args.dataset)
        latency = compute_latency(model)
        params = compute_params_(model)
        param_prune.append(params)
        latency_prune.append(latency)
        ## efficient
        if args.dataset == "CIFAR100":
            fname = "resnet18_10_eff_c100"
        elif args.dataset == "tinyimagenet":
            fname = "resnet18_10_eff_tinyimagenet"
        # low
        new_config = PreActResNet.prepare_filters(PreActResNet, filters, ratio=ratio, neuralscale=True, descent_idx=0, prune_fname=fname, num_classes=num_classes)
        model = PreActResNet18(ratio=ratio, neuralscale=True, num_classes=num_classes, prune_fname=fname, descent_idx=0)
        latency = compute_latency(model)
        params = compute_params_(model)
        param_low.append(params)
        latency_low.append(latency)
        # high
        new_config = PreActResNet.prepare_filters(PreActResNet, filters, ratio=ratio, neuralscale=True, descent_idx=14, prune_fname=fname, num_classes=num_classes)
        model = PreActResNet18(ratio=ratio, neuralscale=True, num_classes=num_classes, prune_fname=fname, descent_idx=14)
        latency = compute_latency(model)
        params = compute_params_(model)
        param_high.append(params)
        latency_high.append(latency)
elif args.model == "mobilenetv2":
    filters = [[32],[16],[24,24],[32,32,32],[64,64,64,64],[96,96,96],[160,160,160],[320],[1280]]
    if args.dataset == "CIFAR100":
        filters_prune = [28, 16, 24, 21, 30, 31, 26, 56, 50, 49, 46, 83, 70, 58, 120, 101, 68, 134, 397] 
        ratios = np.arange(0.25,2.1,0.25) # [0.25, 0.5 , 0.75, 1, 1.25, 1.5 , 1.75, 2]
    elif args.dataset == "tinyimagenet":
        filters_prune = [28, 16, 24, 24, 32, 32, 30, 64, 59, 50, 41, 96, 73, 48, 160, 69, 47, 155, 360] # mobilenetv2 tinyimagenet
        ratios = [0.25,0.5,0.75,1.0]
    for ratio in ratios:
        # convcut
        new_config = MobileNetV2.prepare_filters(MobileNetV2, filters, ratio=ratio, neuralscale=False, num_classes=num_classes)
        model = MobileNetV2(filters=new_config, num_classes=num_classes, dataset=args.dataset, convcut=True)
        latency = compute_latency(model)
        params = compute_params_(model)
        param_convcut.append(params)
        latency_convcut.append(latency)
        # uniform
        new_config = MobileNetV2.prepare_filters(MobileNetV2, filters, ratio=ratio, neuralscale=False, num_classes=num_classes)
        model = MobileNetV2(filters=new_config, num_classes=num_classes, dataset=args.dataset)
        latency = compute_latency(model)
        params = compute_params_(model)
        param_uni.append(params)
        latency_uni.append(latency)
        # pruned
        new_config = MobileNetV2.prepare_filters(MobileNetV2, filters, ratio=ratio, neuralscale=False, num_classes=num_classes, pruned_filters=filters_prune)
        model = MobileNetV2(filters=new_config, num_classes=num_classes, dataset=args.dataset)
        latency = compute_latency(model)
        params = compute_params_(model)
        param_prune.append(params)
        latency_prune.append(latency)
        ## efficient
        if args.dataset == "CIFAR100":
            fname = "mobilenetv2_10_eff_c100"
        elif args.dataset == "tinyimagenet":
            fname = "mobilenetv2_15_eff_tinyimagenet"
        # low
        new_config = MobileNetV2.prepare_filters(MobileNetV2, filters, ratio=ratio, neuralscale=True, descent_idx=0, prune_fname=fname, num_classes=num_classes)
        model = MobileNetV2(ratio=ratio, neuralscale=True, num_classes=num_classes, prune_fname=fname, descent_idx=0)
        latency = compute_latency(model)
        params = compute_params_(model)
        param_low.append(params)
        latency_low.append(latency)
        # high
        new_config = MobileNetV2.prepare_filters(MobileNetV2, filters, ratio=ratio, neuralscale=True, descent_idx=14, prune_fname=fname, num_classes=num_classes)
        model = MobileNetV2(ratio=ratio, neuralscale=True, num_classes=num_classes, prune_fname=fname, descent_idx=14)
        latency = compute_latency(model)
        params = compute_params_(model)
        param_high.append(params)
        latency_high.append(latency)

print("Parameters:")
print("Uniform", param_uni)
print("MorphNet (Taylor-FO)", param_prune)
print("NeuralScale (Iter=1)", param_low)
print("NeuralScale (Iter=15)", param_high)
if args.convcut:
    print("Uniform (ConvCut)", param_convcut)
uni_test_acc = []
prune_uni_test_acc = []
high_test_acc = []
low_test_acc = []
convcut_test_acc = []

uni_test_acc_max = []
prune_uni_test_acc_max = []
high_test_acc_max = []
low_test_acc_max = []
convcut_test_acc_max = []

uni_test_acc_min = []
prune_uni_test_acc_min = []
high_test_acc_min = []
low_test_acc_min = []
convcut_test_acc_min = []

uni_test_acc_std = []
prune_uni_test_acc_std = []
high_test_acc_std = []
low_test_acc_std = []
convcut_test_acc_std = []

if args.dataset=="CIFAR10" or args.dataset=="CIFAR100":
    ratios = np.arange(0.25,2.1,0.25)
elif args.dataset=="tinyimagenet":
    ratios = [0.25,0.5,0.75,1.0]
    
for ratio in ratios:
    uni_test_acc_tmp = []
    prune_uni_test_acc_tmp = []
    high_test_acc_tmp = []
    low_test_acc_tmp = []
    convcut_test_acc_tmp = []
    num_samples = 5
    for i in range(num_samples):
        if args.model == "vgg":
            if args.dataset == "CIFAR100":
                # Baseline (Uniform Scale)
                pkl_ld = pickle.load( open( "saved_plots/vgg/vgg_uni_c100_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
                uni_test_acc_tmp.append(max(pkl_ld["test_acc"]))
                # MorphNet (Taylor-FO)
                pkl_ld = pickle.load( open( "saved_plots/vgg/vgg_pruned_c100_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
                prune_uni_test_acc_tmp.append(max(pkl_ld["test_acc"]))
                # neuralscale (Iteration=15, lr=0.1)
                pkl_ld = pickle.load( open( "saved_plots/vgg/vgg_10_eff_c100_late_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
                high_test_acc_tmp.append(max(pkl_ld["test_acc"]))
                # neuralscale (Iteration=1, lr=0.1)
                pkl_ld = pickle.load( open( "saved_plots/vgg/vgg_10_eff_c100_early_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
                low_test_acc_tmp.append(max(pkl_ld["test_acc"]))
            elif args.dataset == "CIFAR10":
                # Baseline (Uniform Scale)
                pkl_ld = pickle.load( open( "saved_plots/vgg/vgg_uni_c10_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
                uni_test_acc_tmp.append(max(pkl_ld["test_acc"]))
                # MorphNet (Taylor-FO)
                pkl_ld = pickle.load( open( "saved_plots/vgg/vgg_pruned_c10_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
                prune_uni_test_acc_tmp.append(max(pkl_ld["test_acc"]))
                # neuralscale (Iteration=15, lr=0.1)
                pkl_ld = pickle.load( open( "saved_plots/vgg/vgg_10_eff_c10_late_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
                high_test_acc_tmp.append(max(pkl_ld["test_acc"]))
                # neuralscale (Iteration=1, lr=0.1)
                pkl_ld = pickle.load( open( "saved_plots/vgg/vgg_10_eff_c10_early_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
                low_test_acc_tmp.append(max(pkl_ld["test_acc"]))
            else:
                print("Dataset Not Found...")
                exit()
        elif args.model == "resnet18":
            if args.dataset == "CIFAR100":
                # Baseline (Uniform Scale)
                pkl_ld = pickle.load( open( "saved_plots/resnet18/resnet18_uni_c100_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
                uni_test_acc_tmp.append(max(pkl_ld["test_acc"]))
                # MorphNet (Taylor-FO)
                pkl_ld = pickle.load( open( "saved_plots/resnet18/resnet18_pruned_c100_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
                prune_uni_test_acc_tmp.append(max(pkl_ld["test_acc"]))
                # neuralscale (Iteration=15)
                pkl_ld = pickle.load( open( "saved_plots/resnet18/resnet18_10_eff_c100_late_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
                high_test_acc_tmp.append(max(pkl_ld["test_acc"]))
                # neuralscale (Iteration=1)
                pkl_ld = pickle.load( open( "saved_plots/resnet18/resnet18_10_eff_c100_early_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
                low_test_acc_tmp.append(max(pkl_ld["test_acc"]))
            elif args.dataset == "tinyimagenet":
                # Baseline (Uniform Scale)
                pkl_ld = pickle.load( open( "saved_plots/resnet18/resnet18_uni_tinyimagenet_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
                uni_test_acc_tmp.append(max(pkl_ld["test_acc"]))
                # MorphNet (Taylor-FO)
                pkl_ld = pickle.load( open( "saved_plots/resnet18/resnet18_pruned_tinyimagenet_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
                prune_uni_test_acc_tmp.append(max(pkl_ld["test_acc"]))
                # neuralscale (Iteration=15, lr=0.1)
                pkl_ld = pickle.load( open( "saved_plots/resnet18/resnet18_10_eff_tinyimagenet_late_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
                high_test_acc_tmp.append(max(pkl_ld["test_acc"]))
                # neuralscale (Iteration=1, lr=0.1)
                pkl_ld = pickle.load( open( "saved_plots/resnet18/resnet18_10_eff_tinyimagenet_early_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
                low_test_acc_tmp.append(max(pkl_ld["test_acc"]))
                # Uniform (Convcut)
                pkl_ld = pickle.load( open( "saved_plots/resnet18/resnet18_convcut_tinyimagenet_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
                convcut_test_acc_tmp.append(max(pkl_ld["test_acc"]))
            else:
                print("Dataset Not Found...")
        elif args.model == "mobilenetv2":
            if args.dataset == "CIFAR100":
                # Baseline (Uniform Scale)
                pkl_ld = pickle.load( open( "saved_plots/mobilenetv2/mobilenetv2_uni_c100_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
                uni_test_acc_tmp.append(max(pkl_ld["test_acc"]))
                # MorphNet (Taylor-FO)
                pkl_ld = pickle.load( open( "saved_plots/mobilenetv2/mobilenetv2_pruned_c100_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
                prune_uni_test_acc_tmp.append(max(pkl_ld["test_acc"]))
                # neuralscale (Iteration=1)
                pkl_ld = pickle.load( open( "saved_plots/mobilenetv2/mobilenetv2_10_eff_c100_early_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
                low_test_acc_tmp.append(max(pkl_ld["test_acc"]))
                # neuralscale (Iteration=15)
                pkl_ld = pickle.load( open( "saved_plots/mobilenetv2/mobilenetv2_10_eff_c100_late_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
                high_test_acc_tmp.append(max(pkl_ld["test_acc"]))
            elif args.dataset == "tinyimagenet":
                # Baseline (Uniform Scale)
                pkl_ld = pickle.load( open( "saved_plots/mobilenetv2/mobilenetv2_uni_tinyimagenet_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
                uni_test_acc_tmp.append(max(pkl_ld["test_acc"]))
                # MorphNet (Taylor-FO)
                pkl_ld = pickle.load( open( "saved_plots/mobilenetv2/mobilenetv2_pruned_tinyimagenet_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
                prune_uni_test_acc_tmp.append(max(pkl_ld["test_acc"]))
                # neuralscale (Iteration=15, lr=0.1)
                pkl_ld = pickle.load( open( "saved_plots/mobilenetv2/mobilenetv2_15_eff_tinyimagenet_early_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
                low_test_acc_tmp.append(max(pkl_ld["test_acc"]))
                # neuralscale (Iteration=15, lr=0.1)
                pkl_ld = pickle.load( open( "saved_plots/mobilenetv2/mobilenetv2_15_eff_tinyimagenet_late_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
                high_test_acc_tmp.append(max(pkl_ld["test_acc"]))
                # Uniform (Convcut)
                pkl_ld = pickle.load( open( "saved_plots/mobilenetv2/mobilenetv2_convcut_tinyimagenet_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
                convcut_test_acc_tmp.append(max(pkl_ld["test_acc"]))
            else:
                print("Dataset Not Found...")


    uni_test_acc_tmp = np.array(uni_test_acc_tmp)
    prune_uni_test_acc_tmp = np.array(prune_uni_test_acc_tmp)
    high_test_acc_tmp = np.array(high_test_acc_tmp)
    low_test_acc_tmp = np.array(low_test_acc_tmp)
    uni_test_acc_max_tmp = uni_test_acc_tmp.max(axis=0)
    prune_uni_test_acc_max_tmp = prune_uni_test_acc_tmp.max(axis=0)
    high_test_acc_max_tmp = high_test_acc_tmp.max(axis=0)
    low_test_acc_max_tmp = low_test_acc_tmp.max(axis=0)

    uni_test_acc_min_tmp = uni_test_acc_tmp.min(axis=0)
    prune_uni_test_acc_min_tmp = prune_uni_test_acc_tmp.min(axis=0)
    high_test_acc_min_tmp = high_test_acc_tmp.min(axis=0)
    low_test_acc_min_tmp = low_test_acc_tmp.min(axis=0)
    
    uni_test_acc_std_tmp = uni_test_acc_tmp.std(axis=0)
    prune_uni_test_acc_std_tmp = prune_uni_test_acc_tmp.std(axis=0)
    high_test_acc_std_tmp = high_test_acc_tmp.std(axis=0)
    low_test_acc_std_tmp = low_test_acc_tmp.std(axis=0)

    uni_test_acc_tmp = uni_test_acc_tmp.mean(axis=0)
    prune_uni_test_acc_tmp = prune_uni_test_acc_tmp.mean(axis=0)
    high_test_acc_tmp = high_test_acc_tmp.mean(axis=0)
    low_test_acc_tmp = low_test_acc_tmp.mean(axis=0)

    uni_test_acc.append(uni_test_acc_tmp)
    prune_uni_test_acc.append(prune_uni_test_acc_tmp)
    high_test_acc.append(high_test_acc_tmp)
    low_test_acc.append(low_test_acc_tmp)

    uni_test_acc_max.append(uni_test_acc_max_tmp)
    prune_uni_test_acc_max.append(prune_uni_test_acc_max_tmp)
    high_test_acc_max.append(high_test_acc_max_tmp)
    low_test_acc_max.append(low_test_acc_max_tmp)

    uni_test_acc_min.append(uni_test_acc_min_tmp)
    prune_uni_test_acc_min.append(prune_uni_test_acc_min_tmp)
    high_test_acc_min.append(high_test_acc_min_tmp)
    low_test_acc_min.append(low_test_acc_min_tmp)

    uni_test_acc_std.append(uni_test_acc_std_tmp)
    prune_uni_test_acc_std.append(prune_uni_test_acc_std_tmp)
    high_test_acc_std.append(high_test_acc_std_tmp)
    low_test_acc_std.append(low_test_acc_std_tmp)

    if args.convcut:
        convcut_test_acc_tmp = np.array(convcut_test_acc_tmp)
        convcut_test_acc_max_tmp = convcut_test_acc_tmp.max(axis=0)
        convcut_test_acc_min_tmp = convcut_test_acc_tmp.min(axis=0)
        convcut_test_acc_std_tmp = convcut_test_acc_tmp.std(axis=0)
        convcut_test_acc_tmp = convcut_test_acc_tmp.mean(axis=0)
        convcut_test_acc.append(convcut_test_acc_tmp)
        convcut_test_acc_max.append(convcut_test_acc_max_tmp)
        convcut_test_acc_min.append(convcut_test_acc_min_tmp)
        convcut_test_acc_std.append(convcut_test_acc_std_tmp)

uni_test_acc = np.array(uni_test_acc)
prune_uni_test_acc = np.array(prune_uni_test_acc)
high_test_acc = np.array(high_test_acc)
low_test_acc = np.array(low_test_acc)

uni_test_acc_max = np.array(uni_test_acc_max)
prune_uni_test_acc_max = np.array(prune_uni_test_acc_max)
high_test_acc_max = np.array(high_test_acc_max)
low_test_acc_max = np.array(low_test_acc_max)

uni_test_acc_min = np.array(uni_test_acc_min)
prune_uni_test_acc_min = np.array(prune_uni_test_acc_min)
high_test_acc_min = np.array(high_test_acc_min)
low_test_acc_min = np.array(low_test_acc_min)

if args.convcut:
    convcut_test_acc = np.array(convcut_test_acc)
    convcut_test_acc_max = np.array(convcut_test_acc_max)
    convcut_test_acc_min = np.array(convcut_test_acc_min)

# PLOT ACCURACY vs PARAMETERS

if args.convcut or args.plot_normal:
    plt.figure(0, figsize=set_size(colwidth))
else:
    plt.figure(0, figsize=set_size(width,0.235))
# plt.figure(0)
plt.plot(param_high, high_test_acc, marker='o', label="NeuralScale (Iteration=15)")
plt.fill_between(param_high, high_test_acc_min, high_test_acc_max, alpha=0.1)
plt.plot(param_low, low_test_acc, marker='o', label="NeuralScale (Iteration=1)")
plt.fill_between(param_low, low_test_acc_min, low_test_acc_max, alpha=0.1)
plt.plot(param_prune, prune_uni_test_acc, marker='o', label="MorphNet (Taylor-FO)")
plt.fill_between(param_prune, prune_uni_test_acc_min, prune_uni_test_acc_max, alpha=0.1)
plt.plot(param_uni, uni_test_acc, marker='o', label="Uniform Scale (Baseline)")
plt.fill_between(param_uni, uni_test_acc_min, uni_test_acc_max, alpha=0.1)
if args.convcut:
    plt.plot(param_convcut, convcut_test_acc, marker='o', label="Uniform Scale (Convcut)")
    plt.fill_between(param_convcut, convcut_test_acc_min, convcut_test_acc_max, alpha=0.1)
# plt.title("Test Accuracy vs Parameters")
plt.xlabel("Parameters")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.tight_layout()
if args.convcut:
    plt.savefig("savefigs/param_acc_convcut_{}_{}.pdf".format(args.model,args.dataset))
else:
    plt.savefig("savefigs/param_acc_{}_{}.pdf".format(args.model,args.dataset))

if args.convcut or args.plot_normal:
    plt.figure(1, figsize=set_size(colwidth))
else:
    plt.figure(1, figsize=set_size(width,0.235))
# plt.figure(1)
param_spec = np.arange(min(param_uni),max(param_uni),(max(param_uni)-min(param_uni))/20)
param_uni_interp = np.interp(param_spec, param_uni, uni_test_acc)
diff_high = np.interp(param_spec, param_high, high_test_acc) - param_uni_interp
diff_low = np.interp(param_spec, param_low, low_test_acc) - param_uni_interp
diff_prune = np.interp(param_spec, param_prune, prune_uni_test_acc) - param_uni_interp
if args.convcut:
    diff_convcut = np.interp(param_spec, param_convcut, convcut_test_acc) - param_uni_interp
plt.plot(param_spec, diff_high, label="NeuralScale (Iteration=15)")
plt.plot(param_spec, diff_low, label="NeuralScale (Iteration=1)")
plt.plot(param_spec, diff_prune, label="MorphNet (Taylor-FO)")
if args.convcut:
    plt.plot(param_spec, diff_convcut, label="Uniform Scale (Convcut)")
# plt.title("Test Accuracy Gap vs Parameters")
plt.xlabel("Parameters")
plt.ylabel("Accuracy Gap")
plt.legend()
plt.grid()
plt.tight_layout()
# if args.convcut:
#     plt.savefig("savefigs/param_acc_gap_convcut_{}_{}.pdf".format(args.model, args.dataset))
# else:
#     plt.savefig("savefigs/param_acc_gap_{}_{}.pdf".format(args.model, args.dataset))

# PLOT ACCURACY vs latency
if args.convcut or args.plot_normal:
    plt.figure(2, figsize=set_size(colwidth))
else:
    plt.figure(2, figsize=set_size(width,0.235))
# plt.figure(2)
plt.plot(latency_high, high_test_acc, marker='o', label="NeuralScale (Iteration=15)")
plt.fill_between(latency_high, high_test_acc_min, high_test_acc_max, alpha=0.1)
plt.plot(latency_low, low_test_acc, marker='o', label="NeuralScale (Iteration=1)")
plt.fill_between(latency_low, low_test_acc_min, low_test_acc_max, alpha=0.1)
plt.plot(latency_prune, prune_uni_test_acc, marker='o', label="MorphNet (Taylor-FO)")
plt.fill_between(latency_prune, prune_uni_test_acc_min, prune_uni_test_acc_max, alpha=0.1)
plt.plot(latency_uni, uni_test_acc, marker='o', label="Uniform Scale (Baseline)")
plt.fill_between(latency_uni, uni_test_acc_min, uni_test_acc_max, alpha=0.1)
if args.convcut:
    plt.plot(latency_convcut, convcut_test_acc, marker='o', label="Uniform Scale (Convcut)")
    plt.fill_between(latency_convcut, convcut_test_acc_min, convcut_test_acc_max, alpha=0.1)
# plt.title("Test Accuracy vs Latency")
plt.xlabel("Latency (ms)")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.tight_layout()
if args.convcut:
    plt.savefig("savefigs/latency_acc_convcut_{}_{}.pdf".format(args.model, args.dataset))
else:
    plt.savefig("savefigs/latency_acc_{}_{}.pdf".format(args.model, args.dataset))

if args.convcut or args.plot_normal:
    plt.figure(3, figsize=set_size(colwidth))
else:
    plt.figure(3, figsize=set_size(width,0.235))
# plt.figure(3)
latency_spec = np.arange(min(latency_uni),max(latency_uni),(max(latency_uni)-min(latency_uni))/20)
latency_uni_interp = np.interp(latency_spec, latency_uni, uni_test_acc)
diff_high = np.interp(latency_spec, latency_high, high_test_acc) - latency_uni_interp
diff_low = np.interp(latency_spec, latency_low, low_test_acc) - latency_uni_interp
diff_prune = np.interp(latency_spec, latency_prune, prune_uni_test_acc) - latency_uni_interp
if args.convcut:
    diff_convcut = np.interp(latency_spec, latency_convcut, convcut_test_acc) - latency_uni_interp
plt.plot(latency_spec, diff_high, label="NeuralScale (Iteration=15)")
plt.plot(latency_spec, diff_low, label="NeuralScale (Iteration=1)")
plt.plot(latency_spec, diff_prune, label="MorphNet (Taylor-FO)")
if args.convcut:
    plt.plot(latency_spec, diff_convcut, label="Uniform Scale (Convcut)")
# plt.title("Test Accuracy Gap vs Latency")
plt.xlabel("Latency (ms)")
plt.ylabel("Accuracy Gap")
plt.legend()
plt.grid()
plt.tight_layout()
# if args.convcut:
#     plt.savefig("savefigs/latency_acc_gap_convcut_{}_{}.pdf".format(args.model, args.dataset))
# else:
#     plt.savefig("savefigs/latency_acc_gap_{}_{}.pdf".format(args.model, args.dataset))


print("===============")
print("Comparison Table")
print("===============")
print("Param    latency   Accuracy    std")
print("Uniform")
for i, ratio in enumerate(ratios):
    print("{}      {}    {}      {}".format(param_uni[i], latency_uni[i], uni_test_acc[i], uni_test_acc_std[i]))
print("MorphNet (Taylor-FO)")
for i, ratio in enumerate(ratios):
    print("{}      {}    {}      {}".format(param_prune[i], latency_prune[i], prune_uni_test_acc[i], prune_uni_test_acc_std[i]))
print("NeuralScale (Iter=1)")
for i, ratio in enumerate(ratios):
    print("{}      {}    {}      {}".format(param_low[i], latency_low[i], low_test_acc[i], low_test_acc_std[i]))
print("NeuralScale (Iter=15)")
for i, ratio in enumerate(ratios):
    print("{}      {}    {}      {}".format(param_high[i], latency_high[i], high_test_acc[i], high_test_acc_std[i]))
if args.convcut:
    print("Uniform (ConvCut)")
    for i, ratio in enumerate(ratios):
        print("{}      {}    {}      {}".format(param_convcut[i], latency_convcut[i], convcut_test_acc[i], convcut_test_acc_std[i]))

plt.show()
