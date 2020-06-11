'''
Plots accuracy using different pretrain epochs as shown in supplementary section
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

parser = argparse.ArgumentParser(description='Plots accuracy using different pretrain epochs')
parser.add_argument('--dataset', default="CIFAR10", type=str,
                        help='dataset for experiment, choice: CIFAR10, CIFAR100', choices= ["CIFAR10", "CIFAR100"])
parser.add_argument('--model', default="vgg", type=str,
                    help='model selection, choices: vgg, mobilenetv2',
                    choices=["vgg", "mobilenetv2"])

args = parser.parse_args()


width = 237.13594 # columnwidth CVPR2020

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
    golden_ratio = (5**.5 - 0.5) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


fig_width, fig_height = set_size(width)


# Settings for plot fonts
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


param_uni = []
param_0 = []
param_2 = []
param_5 = []
param_10 = []
param_30 = []
param_60 = []
latency_uni = []
latency_0 = []
latency_2 = []
latency_5 = []
latency_10 = []
latency_30 = []
latency_60 = []
if args.model=="vgg": # CIFAR10 only
    config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
    ratios = np.arange(0.25,2.1,0.25) # [0.25, 0.5 , 0.75, 1, 1.25, 1.5 , 1.75, 2]
    for ratio in ratios:
        # uniform
        new_config = VGG.prepare_filters(VGG, config, ratio=ratio, neuralscale=False, num_classes=num_classes)
        model = vgg11(config=new_config, num_classes=num_classes)
        latency = compute_latency(model)
        params = compute_params_(model)
        param_uni.append(params)
        latency_uni.append(latency)
        ## efficient
        vgg_0_fname = "vgg_0_eff_c10"
        vgg_2_fname = "vgg_2_eff_c10"
        vgg_5_fname = "vgg_5_eff_c10"
        vgg_10_fname = "vgg_10_eff_c10"
        vgg_30_fname = "vgg_30_eff_c10"
        vgg_60_fname = "vgg_60_eff_c10"
        # P=0
        new_config = VGG.prepare_filters(VGG, config, ratio=ratio, neuralscale=True, descent_idx=14, prune_fname=vgg_0_fname, num_classes=num_classes)
        model = vgg11(ratio=ratio, neuralscale=True, num_classes=num_classes, prune_fname=vgg_0_fname, descent_idx=14)
        latency = compute_latency(model)
        params = compute_params_(model)
        param_0.append(params)
        latency_0.append(latency)
        # P=2
        new_config = VGG.prepare_filters(VGG, config, ratio=ratio, neuralscale=True, descent_idx=14, prune_fname=vgg_2_fname, num_classes=num_classes)
        model = vgg11(ratio=ratio, neuralscale=True, num_classes=num_classes, prune_fname=vgg_2_fname, descent_idx=14)
        latency = compute_latency(model)
        params = compute_params_(model)
        param_2.append(params)
        latency_2.append(latency)
        # P=5
        new_config = VGG.prepare_filters(VGG, config, ratio=ratio, neuralscale=True, descent_idx=14, prune_fname=vgg_5_fname, num_classes=num_classes)
        model = vgg11(ratio=ratio, neuralscale=True, num_classes=num_classes, prune_fname=vgg_5_fname, descent_idx=14)
        latency = compute_latency(model)
        params = compute_params_(model)
        param_5.append(params)
        latency_5.append(latency)
        # P=10
        new_config = VGG.prepare_filters(VGG, config, ratio=ratio, neuralscale=True, descent_idx=14, prune_fname=vgg_10_fname, num_classes=num_classes)
        model = vgg11(ratio=ratio, neuralscale=True, num_classes=num_classes, prune_fname=vgg_10_fname, descent_idx=14)
        latency = compute_latency(model)
        params = compute_params_(model)
        param_10.append(params)
        latency_10.append(latency)
        # P=30
        new_config = VGG.prepare_filters(VGG, config, ratio=ratio, neuralscale=True, descent_idx=14, prune_fname=vgg_30_fname, num_classes=num_classes)
        model = vgg11(ratio=ratio, neuralscale=True, num_classes=num_classes, prune_fname=vgg_30_fname, descent_idx=14)
        latency = compute_latency(model)
        params = compute_params_(model)
        param_30.append(params)
        latency_30.append(latency)
        # P=60
        new_config = VGG.prepare_filters(VGG, config, ratio=ratio, neuralscale=True, descent_idx=14, prune_fname=vgg_60_fname, num_classes=num_classes)
        model = vgg11(ratio=ratio, neuralscale=True, num_classes=num_classes, prune_fname=vgg_60_fname, descent_idx=14)
        latency = compute_latency(model)
        params = compute_params_(model)
        param_60.append(params)
        latency_60.append(latency)
elif args.model == "mobilenetv2": # CIFAR100 only
    filters = [[32],[16],[24,24],[32,32,32],[64,64,64,64],[96,96,96],[160,160,160],[320],[1280]]
    ratios = np.arange(0.25,2.1,0.25) # [0.25, 0.5 , 0.75, 1, 1.25, 1.5 , 1.75, 2]
    for ratio in ratios:
        # uniform
        new_config = MobileNetV2.prepare_filters(MobileNetV2, filters, ratio=ratio, neuralscale=False, num_classes=num_classes)
        model = MobileNetV2(filters=new_config, num_classes=num_classes, dataset=args.dataset)
        latency = compute_latency(model)
        params = compute_params_(model)
        param_uni.append(params)
        latency_uni.append(latency)
        ## efficient
        mobilenetv2_0_fname = "mobilenetv2_0_eff_c100"
        mobilenetv2_2_fname = "mobilenetv2_2_eff_c100"
        mobilenetv2_5_fname = "mobilenetv2_5_eff_c100"
        mobilenetv2_10_fname = "mobilenetv2_10_eff_c100"
        mobilenetv2_30_fname = "mobilenetv2_30_eff_c100"
        mobilenetv2_60_fname = "mobilenetv2_60_eff_c100"
        # P=0
        new_config = MobileNetV2.prepare_filters(MobileNetV2, filters, ratio=ratio, neuralscale=True, descent_idx=13, prune_fname=mobilenetv2_0_fname, num_classes=num_classes)
        model = MobileNetV2(ratio=ratio, neuralscale=True, num_classes=num_classes, prune_fname=mobilenetv2_0_fname, descent_idx=13)
        latency = compute_latency(model)
        params = compute_params_(model)
        param_0.append(params)
        latency_0.append(latency)
        # P=2
        new_config = MobileNetV2.prepare_filters(MobileNetV2, filters, ratio=ratio, neuralscale=True, descent_idx=14, prune_fname=mobilenetv2_2_fname, num_classes=num_classes)
        model = MobileNetV2(ratio=ratio, neuralscale=True, num_classes=num_classes, prune_fname=mobilenetv2_2_fname, descent_idx=14)
        latency = compute_latency(model)
        params = compute_params_(model)
        param_2.append(params)
        latency_2.append(latency)
        # P=5
        new_config = MobileNetV2.prepare_filters(MobileNetV2, filters, ratio=ratio, neuralscale=True, descent_idx=14, prune_fname=mobilenetv2_5_fname, num_classes=num_classes)
        model = MobileNetV2(ratio=ratio, neuralscale=True, num_classes=num_classes, prune_fname=mobilenetv2_5_fname, descent_idx=14)
        latency = compute_latency(model)
        params = compute_params_(model)
        param_5.append(params)
        latency_5.append(latency)
        # P=10
        new_config = MobileNetV2.prepare_filters(MobileNetV2, filters, ratio=ratio, neuralscale=True, descent_idx=14, prune_fname=mobilenetv2_10_fname, num_classes=num_classes)
        model = MobileNetV2(ratio=ratio, neuralscale=True, num_classes=num_classes, prune_fname=mobilenetv2_10_fname, descent_idx=14)
        latency = compute_latency(model)
        params = compute_params_(model)
        param_10.append(params)
        latency_10.append(latency)
        # P=30
        new_config = MobileNetV2.prepare_filters(MobileNetV2, filters, ratio=ratio, neuralscale=True, descent_idx=14, prune_fname=mobilenetv2_30_fname, num_classes=num_classes)
        model = MobileNetV2(ratio=ratio, neuralscale=True, num_classes=num_classes, prune_fname=mobilenetv2_30_fname, descent_idx=14)
        latency = compute_latency(model)
        params = compute_params_(model)
        param_30.append(params)
        latency_30.append(latency)
        # # P=60
        new_config = MobileNetV2.prepare_filters(MobileNetV2, filters, ratio=ratio, neuralscale=True, descent_idx=13, prune_fname=mobilenetv2_60_fname, num_classes=num_classes)
        model = MobileNetV2(ratio=ratio, neuralscale=True, num_classes=num_classes, prune_fname=mobilenetv2_60_fname, descent_idx=14)
        latency = compute_latency(model)
        params = compute_params_(model)
        param_60.append(params)
        latency_60.append(latency)
        
print(param_uni)
print(param_0)
print(param_2)
print(param_5)
print(param_10)
print(param_30)
print(param_60)

test_acc = []
p0_test_acc = []
p2_test_acc = []
p5_test_acc = []
p10_test_acc = []
p30_test_acc = []
p60_test_acc = []
test_acc_max = []
p0_test_acc_max = []
p2_test_acc_max = []
p5_test_acc_max = []
p10_test_acc_max = []
p30_test_acc_max = []
p60_test_acc_max = []
p0_test_acc_min = []
test_acc_min = []
p2_test_acc_min = []
p5_test_acc_min = []
p10_test_acc_min = []
p30_test_acc_min = []
p60_test_acc_min = []
test_acc_std = []
p0_test_acc_std = []
p2_test_acc_std = []
p5_test_acc_std = []
p10_test_acc_std = []
p30_test_acc_std = []
p60_test_acc_std = []

ratios = np.arange(0.25,2.1,0.25)
for ratio in ratios:
    test_acc_tmp = []
    p0_test_acc_tmp = []
    p2_test_acc_tmp = []
    p5_test_acc_tmp = []
    p10_test_acc_tmp = []
    p30_test_acc_tmp = []
    p60_test_acc_tmp = []
    for i in range(5):
        if args.model == "vgg":
            # Uniform
            pkl_ld = pickle.load( open( "saved_plots/vgg/vgg_uni_c10_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
            test_acc_tmp.append(max(pkl_ld["test_acc"]))
            # P=0
            pkl_ld = pickle.load( open( "saved_plots/vgg/vgg_0_eff_c10_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
            p0_test_acc_tmp.append(max(pkl_ld["test_acc"]))
            # P=2
            pkl_ld = pickle.load( open( "saved_plots/vgg/vgg_2_eff_c10_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
            p2_test_acc_tmp.append(max(pkl_ld["test_acc"]))
            # P=5
            pkl_ld = pickle.load( open( "saved_plots/vgg/vgg_5_eff_c10_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
            p5_test_acc_tmp.append(max(pkl_ld["test_acc"]))
            # P=10
            pkl_ld = pickle.load( open( "saved_plots/vgg/vgg_10_eff_c10_late_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
            p10_test_acc_tmp.append(max(pkl_ld["test_acc"]))
            # P=30
            pkl_ld = pickle.load( open( "saved_plots/vgg/vgg_30_eff_c10_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
            p30_test_acc_tmp.append(max(pkl_ld["test_acc"]))
            # P=60
            pkl_ld = pickle.load( open( "saved_plots/vgg/vgg_60_eff_c10_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
            p60_test_acc_tmp.append(max(pkl_ld["test_acc"]))
        elif args.model == "mobilenetv2":
            # Uniform
            pkl_ld = pickle.load( open( "saved_plots/mobilenetv2/mobilenetv2_uni_c100_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
            test_acc_tmp.append(max(pkl_ld["test_acc"]))
            # P=0
            pkl_ld = pickle.load( open( "saved_plots/mobilenetv2/mobilenetv2_0_eff_c100_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
            p0_test_acc_tmp.append(max(pkl_ld["test_acc"]))
            # P=2
            pkl_ld = pickle.load( open( "saved_plots/mobilenetv2/mobilenetv2_2_eff_c100_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
            p2_test_acc_tmp.append(max(pkl_ld["test_acc"]))
            # P=5
            pkl_ld = pickle.load( open( "saved_plots/mobilenetv2/mobilenetv2_5_eff_c100_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
            p5_test_acc_tmp.append(max(pkl_ld["test_acc"]))
            # P=10
            pkl_ld = pickle.load( open( "saved_plots/mobilenetv2/mobilenetv2_10_eff_c100_late_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
            p10_test_acc_tmp.append(max(pkl_ld["test_acc"]))
            # P=30
            pkl_ld = pickle.load( open( "saved_plots/mobilenetv2/mobilenetv2_30_eff_c100_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
            p30_test_acc_tmp.append(max(pkl_ld["test_acc"]))
            # # P=60
            pkl_ld = pickle.load( open( "saved_plots/mobilenetv2/mobilenetv2_60_eff_c100_{}_{}.pk".format(i,int(ratio*100)), "rb" ) )
            p60_test_acc_tmp.append(max(pkl_ld["test_acc"]))
           
    test_acc_tmp = np.array(test_acc_tmp)
    p0_test_acc_tmp = np.array(p0_test_acc_tmp)
    p2_test_acc_tmp = np.array(p2_test_acc_tmp)
    p5_test_acc_tmp = np.array(p5_test_acc_tmp)
    p10_test_acc_tmp = np.array(p10_test_acc_tmp)
    p30_test_acc_tmp = np.array(p30_test_acc_tmp)
    p60_test_acc_tmp = np.array(p60_test_acc_tmp)

    test_acc_max_tmp = test_acc_tmp.max(axis=0)    
    p0_test_acc_max_tmp = p0_test_acc_tmp.max(axis=0)
    p2_test_acc_max_tmp = p2_test_acc_tmp.max(axis=0)
    p5_test_acc_max_tmp = p5_test_acc_tmp.max(axis=0)
    p10_test_acc_max_tmp = p10_test_acc_tmp.max(axis=0)
    p30_test_acc_max_tmp = p30_test_acc_tmp.max(axis=0)
    p60_test_acc_max_tmp = p60_test_acc_tmp.max(axis=0)

    test_acc_min_tmp = test_acc_tmp.min(axis=0)    
    p0_test_acc_min_tmp = p0_test_acc_tmp.min(axis=0)
    p2_test_acc_min_tmp = p2_test_acc_tmp.min(axis=0)
    p5_test_acc_min_tmp = p5_test_acc_tmp.min(axis=0)
    p10_test_acc_min_tmp = p10_test_acc_tmp.min(axis=0)
    p30_test_acc_min_tmp = p30_test_acc_tmp.min(axis=0)
    p60_test_acc_min_tmp = p60_test_acc_tmp.min(axis=0)
    
    test_acc_std_tmp = test_acc_tmp.std(axis=0)
    p0_test_acc_std_tmp = p0_test_acc_tmp.std(axis=0)
    p2_test_acc_std_tmp = p2_test_acc_tmp.std(axis=0)
    p5_test_acc_std_tmp = p5_test_acc_tmp.std(axis=0)
    p10_test_acc_std_tmp = p10_test_acc_tmp.std(axis=0)
    p30_test_acc_std_tmp = p30_test_acc_tmp.std(axis=0)
    p60_test_acc_std_tmp = p60_test_acc_tmp.std(axis=0)
    
    test_acc_tmp = test_acc_tmp.mean(axis=0)
    p0_test_acc_tmp = p0_test_acc_tmp.mean(axis=0)
    p2_test_acc_tmp = p2_test_acc_tmp.mean(axis=0)
    p5_test_acc_tmp = p5_test_acc_tmp.mean(axis=0)
    p10_test_acc_tmp = p10_test_acc_tmp.mean(axis=0)
    p30_test_acc_tmp = p30_test_acc_tmp.mean(axis=0)
    p60_test_acc_tmp = p60_test_acc_tmp.mean(axis=0)


    test_acc.append(test_acc_tmp)    
    p0_test_acc.append(p0_test_acc_tmp)
    p2_test_acc.append(p2_test_acc_tmp)
    p5_test_acc.append(p5_test_acc_tmp)
    p10_test_acc.append(p10_test_acc_tmp)
    p30_test_acc.append(p30_test_acc_tmp)
    p60_test_acc.append(p60_test_acc_tmp)

    test_acc_max.append(test_acc_max_tmp)    
    p0_test_acc_max.append(p0_test_acc_max_tmp)
    p2_test_acc_max.append(p2_test_acc_max_tmp)
    p5_test_acc_max.append(p5_test_acc_max_tmp)
    p10_test_acc_max.append(p10_test_acc_max_tmp)
    p30_test_acc_max.append(p30_test_acc_max_tmp)
    p60_test_acc_max.append(p60_test_acc_max_tmp)

    test_acc_min.append(test_acc_min_tmp)    
    p0_test_acc_min.append(p0_test_acc_min_tmp)
    p2_test_acc_min.append(p2_test_acc_min_tmp)
    p5_test_acc_min.append(p5_test_acc_min_tmp)
    p10_test_acc_min.append(p10_test_acc_min_tmp)
    p30_test_acc_min.append(p30_test_acc_min_tmp)
    p60_test_acc_min.append(p60_test_acc_min_tmp)

    test_acc_std.append(test_acc_std_tmp)
    p0_test_acc_std.append(p0_test_acc_std_tmp)
    p2_test_acc_std.append(p2_test_acc_std_tmp)
    p5_test_acc_std.append(p5_test_acc_std_tmp)
    p10_test_acc_std.append(p10_test_acc_std_tmp)
    p30_test_acc_std.append(p30_test_acc_std_tmp)
    p60_test_acc_std.append(p60_test_acc_std_tmp)


test_acc = np.array(test_acc)
p0_test_acc = np.array(p0_test_acc)
p2_test_acc = np.array(p2_test_acc)
p5_test_acc = np.array(p5_test_acc)
p10_test_acc = np.array(p10_test_acc)
p30_test_acc = np.array(p30_test_acc)
p60_test_acc = np.array(p60_test_acc)

test_acc_max = np.array(test_acc_max)
p0_test_acc_max = np.array(p0_test_acc_max)
p2_test_acc_max = np.array(p2_test_acc_max)
p5_test_acc_max = np.array(p5_test_acc_max)
p10_test_acc_max = np.array(p10_test_acc_max)
p30_test_acc_max = np.array(p30_test_acc_max)
p60_test_acc_max = np.array(p60_test_acc_max)

test_acc_min = np.array(test_acc_min)
p0_test_acc_min = np.array(p0_test_acc_min)
p2_test_acc_min = np.array(p2_test_acc_min)
p5_test_acc_min = np.array(p5_test_acc_min)
p10_test_acc_min = np.array(p10_test_acc_min)
p30_test_acc_min = np.array(p30_test_acc_min)
p60_test_acc_min = np.array(p60_test_acc_min)

test_acc_std = np.array(test_acc_std)
p0_test_acc_std = np.array(p0_test_acc_std)
p2_test_acc_std = np.array(p2_test_acc_std)
p5_test_acc_std = np.array(p5_test_acc_std)
p10_test_acc_std = np.array(p10_test_acc_std)
p30_test_acc_std = np.array(p30_test_acc_std)
p60_test_acc_std = np.array(p60_test_acc_std)

plt.figure(1, figsize=set_size(width))
plt.plot(param_0, p0_test_acc, marker='o', label=r"$P=0$")
plt.fill_between(param_0, p0_test_acc_min, p0_test_acc_max, alpha=0.1)
plt.plot(param_2, p2_test_acc, marker='o', label=r"$P=2$")
plt.fill_between(param_2, p2_test_acc_min, p2_test_acc_max, alpha=0.1)
plt.plot(param_5, p5_test_acc, marker='o', label=r"$P=5$")
plt.fill_between(param_5, p5_test_acc_min, p5_test_acc_max, alpha=0.1)
plt.plot(param_10, p10_test_acc, marker='o', label=r"$P=10$")
plt.fill_between(param_10, p10_test_acc_min, p10_test_acc_max, alpha=0.1)
plt.plot(param_30, p30_test_acc, marker='o', label=r"$P=30$")
plt.fill_between(param_30, p30_test_acc_min, p30_test_acc_max, alpha=0.1)
plt.plot(param_60, p60_test_acc, marker='o', label=r"$P=60$")
plt.fill_between(param_60, p60_test_acc_min, p60_test_acc_max, alpha=0.1)
plt.plot(param_uni, test_acc, marker='o', color='black', linestyle='dashed', label="Uniform Scale (Baseline)")
# plt.title("Test Accuracy vs Parameters")
plt.xlabel("Parameters")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("savefigs/param_pretrain_{}_{}.pdf".format(args.model,args.dataset))


plt.figure(2, figsize=set_size(width))
plt.plot(latency_0, p0_test_acc, marker='o', label=r"$P=0$")
plt.fill_between(latency_0, p0_test_acc_min, p0_test_acc_max, alpha=0.1)
plt.plot(latency_2, p2_test_acc, marker='o', label=r"$P=2$")
plt.fill_between(latency_2, p2_test_acc_min, p2_test_acc_max, alpha=0.1)
plt.plot(latency_5, p5_test_acc, marker='o', label=r"$P=5$")
plt.fill_between(latency_5, p5_test_acc_min, p5_test_acc_max, alpha=0.1)
plt.plot(latency_10, p10_test_acc, marker='o', label=r"$P=10$")
plt.fill_between(latency_10, p10_test_acc_min, p10_test_acc_max, alpha=0.1)
plt.plot(latency_30, p30_test_acc, marker='o', label=r"$P=30$")
plt.fill_between(latency_30, p30_test_acc_min, p30_test_acc_max, alpha=0.1)
plt.plot(latency_60, p60_test_acc, marker='o', label=r"$P=60$")
plt.fill_between(latency_60, p60_test_acc_min, p60_test_acc_max, alpha=0.1)
plt.plot(latency_uni, test_acc, marker='o', color='black', linestyle='dashed', label="Uniform Scale (Baseline)")
# plt.title("Test Accuracy vs Latency")
plt.xlabel("Latency (ms)")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("savefigs/latency_pretrain_{}_{}.pdf".format(args.model,args.dataset))


print("===============")
print("Comparison Table")
print("===============")
print("Param    latency   Accuracy    std")
print("Baseline")
for i, ratio in enumerate(ratios):
    print("{}      {}    {}      {}".format(param_uni[i], latency_uni[i], test_acc[i], test_acc_std[i]))
print("P=0")
for i, ratio in enumerate(ratios):
    print("{}      {}    {}      {}".format(param_0[i], latency_0[i], p0_test_acc[i], p0_test_acc_std[i]))
print("P=2")
for i, ratio in enumerate(ratios):
    print("{}      {}    {}      {}".format(param_2[i], latency_2[i], p2_test_acc[i], p2_test_acc_std[i]))
print("P=5")
for i, ratio in enumerate(ratios):
    print("{}      {}    {}      {}".format(param_5[i], latency_5[i], p5_test_acc[i], p5_test_acc_std[i]))
print("P=10")
for i, ratio in enumerate(ratios):
    print("{}      {}    {}      {}".format(param_10[i], latency_10[i], p10_test_acc[i], p10_test_acc_std[i]))
print("P=30")
for i, ratio in enumerate(ratios):
    print("{}      {}    {}      {}".format(param_30[i], latency_30[i], p30_test_acc[i], p30_test_acc_std[i]))
print("P=60")
for i, ratio in enumerate(ratios):
    print("{}      {}    {}      {}".format(param_60[i], latency_60[i], p60_test_acc[i], p60_test_acc_std[i]))

plt.show()
