'''
List configuration obtained using architecture descent for different parameter sizes
'''

import pickle
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pylab as plt
import argparse
from utils import compute_params
from model.VGG import VGG
from model.preact_resnet import PreActResNet
from model.mobilenetv2 import MobileNetV2

parser = argparse.ArgumentParser(description='List configuration obtained using architecture descent for different parameter sizes')
parser.add_argument('--model', default="resnet18", type=str,
                    help='model selection, choices: vgg, mobilenetv2, resnet18',
                    choices=["vgg", "mobilenetv2", "resnet18"])
parser.add_argument('--prune_fname', default='filename',
                    help='prune save file')
parser.add_argument('--plot', dest="plot", action='store_true', default=False,
                    help='Show plot of architecture descent')
parser.add_argument('--dataset', default="CIFAR10", type=str,
                        help='dataset for experiment, choice: CIFAR10, CIFAR100, tinyimagenet', choices= ["CIFAR10", "CIFAR100", "tinyimagenet"])
parser.add_argument('--pretrain', type=int, default=None,
                    help='number of warm-up or fine-tuning epochs before pruning (default: None)')
args = parser.parse_args()

# Settings for plot fonts
nice_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "Times New Roman",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 6,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
}

mpl.rcParams.update(nice_fonts)
sns.set_context("paper", rc=nice_fonts)



if args.dataset == 'CIFAR10':
    num_classes = 10
elif args.dataset == 'CIFAR100':
    num_classes = 100
elif args.dataset == "tinyimagenet":
    num_classes = 200

plot_growth = args.plot

if plot_growth:
    # width = 496.85625/2
    width = 237.13594
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
        golden_ratio = (5**.5 - 0.75) / 2
        # golden_ratio = 0.5

        # Figure width in inches
        fig_width_in = fig_width_pt * inches_per_pt
        # Figure height in inches
        fig_height_in = fig_width_in * golden_ratio

        fig_dim = (fig_width_in, fig_height_in)

        return fig_dim
    fig, axs = plt.subplots(2, 2, figsize=set_size(width), sharex=True, sharey=True)
    # fig, axs = plt.subplots(1, 4, figsize=set_size(width), sharex=True, sharey=True)
    # fig, axs = plt.subplots(1, 4, figsize=(25,4.5))

if args.dataset == "tinyimagenet":
    # ratios = [0.25, 1.0]
    ratios = [0.25, 0.5, 0.75, 1.0]
else:
    # ratios = [0.25, 2]
    ratios = [0.25, 0.75, 1.25, 2]
if args.model == "vgg":
    original_filters = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
elif args.model == "resnet18":
    original_filters = [[64],[64,64],[64,64],[128,128],[128,128],[256,256],[256,256],[512,512],[512,512]]
elif args.model == "mobilenetv2":
    original_filters = [[32],[16],[24,24],[32,32,32],[64,64,64,64],[96,96,96],[160,160,160],[320],[1280]]

for idx, ratio in enumerate(ratios):
    if plot_growth:
        total_params = []
    print("Ratio: {}".format(ratio))
    for iteration in range(15):
        if args.model == "vgg":
            filters = VGG.prepare_filters(VGG, original_filters, ratio=ratio, neuralscale=True, num_classes=num_classes, prune_fname=args.prune_fname, descent_idx=iteration)
            filters = [cfg for cfg in list(filter(lambda a: a != 'M', filters))]
        elif args.model == "resnet18":
            filters = PreActResNet.prepare_filters(PreActResNet, original_filters, ratio=ratio, neuralscale=True, num_classes=num_classes, prune_fname=args.prune_fname, descent_idx=iteration)
            filters = [cfg for cfg in sum(filters,[])]
        elif args.model == "mobilenetv2":
            filters = MobileNetV2.prepare_filters(MobileNetV2, original_filters, ratio=ratio, neuralscale=True, num_classes=num_classes, prune_fname=args.prune_fname, descent_idx=iteration)
            filters = [cfg for cfg in sum(filters,[])][:-2]
        if plot_growth:
            total_params.append(filters)
                
    if plot_growth:
        # ax = sns.heatmap(np.array(total_params).T, linewidth=0.005, xticklabels=2, yticklabels=2, ax=axs[idx])
        ax = sns.heatmap(np.array(total_params).T, linewidth=0.005, xticklabels=2, ax=axs[int(idx/2)][idx%2])
        ax.set_title("Ratio={}".format(ratio))
        # if idx!=0:
        #     ax.tick_params(axis='y', which='both', width=0)
        if idx==1 or idx==3:
            ax.tick_params(axis='y', which='both', width=0, length=0)
        if idx==0 or idx==1:
            ax.tick_params(axis='x', which='both', width=0, length=0)

if plot_growth:
    # Bug in current version of sns heatmap
    b, t = plt.ylim()
    # for i in range(2):
    for i in range(4):
        axs[int(i/2)][i%2].set_ylim(b+0.5, t-0.5)
        # axs[i].set_ylim(b+0.5, t-0.5)

    # for ax in axs.flat:
    #     ax.set(xlabel='Iteration', ylabel='Layer')
    # for ax in axs.flat:
    #     ax.label_outer()
    # plt.suptitle("Architecture Descent")
    fig.text(0.5, 0.01, "Iteration", ha='center')
    fig.text(0.01, 0.5, "Layer", va='center', rotation='vertical')
    plt.tight_layout()
    if args.pretrain == None:
        plt.savefig("savefigs/architecture_{}_{}.pdf".format(args.model,args.dataset))
    else:
        plt.savefig("savefigs/architecture_{}_{}_{}.pdf".format(args.pretrain,args.model,args.dataset))
    
    plt.show()
