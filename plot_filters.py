'''
Plots residual filters obtained via iterative pruning
'''

import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pylab as plt
from sklearn import linear_model
import pickle
from cvxopt import matrix, solvers
import argparse
from utils import compute_params

parser = argparse.ArgumentParser(description='Efficient Filter Scaling of Convolutional Neural Network')
parser.add_argument('--model', default="resnet18", type=str,
                    help='model selection, choices: vgg, mobilenetv2, resnet18',
                    choices=["vgg", "mobilenetv2", "resnet18"])
parser.add_argument('--prune_fname', default='filename',
                    help='prune save file')
parser.add_argument('--dataset', default="CIFAR10", type=str,
                        help='dataset for experiment, choice: CIFAR10, CIFAR100, tinyimagenet', choices= ["CIFAR10", "CIFAR100", "tinyimagenet"])
args = parser.parse_args()


# Settings for plot fonts
nice_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        # "font.family": "Times New Roman",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.linewidth": 4 / 12.,
        "lines.linewidth": 4 / 12.,
        "patch.linewidth": 4 / 12.,
        "axes.labelsize": 4,
        "font.size": 4,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 2,
        "xtick.labelsize": 3,
        "ytick.labelsize": 3,
}

mpl.rcParams.update(nice_fonts)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']


if args.dataset == 'CIFAR10':
    num_classes = 10
elif args.dataset == 'CIFAR100':
    num_classes = 100
elif args.dataset == 'tinyimagenet':
    num_classes = 200

solvers.options['show_progress'] = False
cmap = plt.get_cmap('tab10')

fig1, axs1 = plt.subplots(2, 1)
fig2, axs2 = plt.subplots(2, 1)

# #####
# EARLY
# #####
f = open('prune_record/' + args.prune_fname + '_0.csv', newline='')
reader = csv.reader(f, delimiter=',')
filters = []
for row in reader:
    filters.append(list(map(int,row)))
filters = np.array(filters)

# Compute total parameters
total_params = []
for filt in filters: # over all iterations
    total_params.append(compute_params(filt, classes=num_classes, model=args.model))
total_params = np.array(total_params)

lin_reg = linear_model.LinearRegression()


ln_k = np.log(total_params)
A = np.stack((ln_k, np.ones(ln_k.shape)), axis=1) 
b = np.log(filters)
x = np.matmul(np.matmul(np.linalg.inv( np.matmul(A.transpose(), A)), A.transpose()), b)
beta = x[0,:]
alpha = np.exp(x[1,:])
filt = np.array([total_params ** b for b in beta])
filt = np.multiply(filt.transpose(), alpha).transpose()

print('early')
for i in range(filters.shape[1]):
    axs1[0].plot(total_params, filters[:,i], label="Layer {}".format(i+1), color=cmap(float(i)/filters.shape[1]))
    axs1[0].plot(total_params, filt[i], label="Layer {} (fit)".format(i+1), linestyle="--", color=cmap(float(i)/filters.shape[1]))
    print(i+1, alpha[i], beta[i])

axs1[0].set_title("Layer's Filter vs Total Parameter (early)")
axs1[0].set(xlabel="Total Parameters", ylabel="Filters")
axs1[0].legend()
f.close()

plt.figure(4)
for i in range(filters.shape[1]):
    plt.plot(total_params, filters[:,i], label="Layer {}".format(i+1), color=cmap(float(i)/filters.shape[1]))
    plt.plot(total_params, filt[i], label="Layer {} (fit)".format(i+1), linestyle="--", color=cmap(float(i)/filters.shape[1]))
    print(i+1, alpha[i], beta[i])

plt.title("Layer's Filter vs Total Parameter (early)")
plt.xlabel("Total Parameters")
plt.ylabel("Filters")
plt.grid()
plt.legend()

# ######
total_params = np.arange(1000,100000000,5000)

filt = np.array([total_params ** b for b in beta])
filt = np.multiply(filt.transpose(), alpha).transpose()
print(filt[0])
for i in range(filters.shape[1]):
    axs2[0].plot(total_params, filt[i], label="Layer {} (fit)".format(i+1), linestyle="dashed", color=cmap(float(i)/filters.shape[1]))

axs2[0].set_title("Experiment (Early)")
axs2[0].set(xlabel="Total Parameters", ylabel="Filters")
axs2[0].legend()

# ####
# LATE
# ####
f = open('prune_record/' + args.prune_fname + '_14.csv', newline='')
reader = csv.reader(f, delimiter=',')
filters = []
for row in reader:
    filters.append(list(map(int,row)))
filters = np.array(filters)

# Compute total parameters
total_params = []
for filt in filters: # over all iterations
    total_params.append(compute_params(filt, classes=num_classes, model=args.model))
total_params = np.array(total_params)

lin_reg = linear_model.LinearRegression()

ln_filt = np.log(filters)
ln_k = np.log(total_params)
A = np.stack((ln_k, np.ones(ln_k.shape)), axis=1) 
b = np.log(filters[:])
x = np.matmul(np.matmul(np.linalg.inv( np.matmul(A.transpose(), A)), A.transpose()), b)
beta = x[0,:]
alpha = np.exp(x[1,:])
filt = np.array([total_params ** b for b in beta])
filt = np.multiply(filt.transpose(), alpha).transpose()

print('late')
for i in range(filters.shape[1]):
    axs1[1].plot(total_params, filters[:,i], label="Layer {}".format(i+1), color=cmap(float(i)/filters.shape[1]))
    axs1[1].plot(total_params, filt[i], label="Layer {} (fit)".format(i+1), linestyle="dashed", color=cmap(float(i)/filters.shape[1]))

axs1[1].set_title("Layer's Filter vs Total Parameter (late)")
axs1[1].set(xlabel="Total Parameters", ylabel="Filters")
axs1[1].legend()
f.close()



# width = 496.85625 # textwidth of CVPR2020
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
    golden_ratio = (5**.5 - .9) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


plt.figure(5, figsize=set_size(width/2))
for i in range(filters.shape[1]):
    plt.plot(total_params, filters[:,i], label="Layer {}".format(i+1), color=cmap(float(i)/filters.shape[1]))
    plt.plot(total_params, filt[i], label="Layer {} (fit)".format(i+1), linestyle="dashed", color=cmap(float(i)/filters.shape[1]))

plt.locator_params(axis='x', nbins=5)
# plt.title("Layer's Filter vs Total Parameter")
plt.xlabel("Total Parameters")
plt.ylabel("Filters")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("savefigs/growth_{}_{}.pdf".format(args.model,args.dataset))

# ######
total_params = np.arange(1000,100000000,5000)

filt = np.array([total_params ** b for b in beta])
filt = np.multiply(filt.transpose(), alpha).transpose()
print(filt[0])
for i in range(filters.shape[1]):
    axs2[1].plot(total_params, filt[i], label="Layer {} (fit)".format(i+1), linestyle="dashed", color=cmap(float(i)/filters.shape[1]))

axs2[1].set_title("Experiment (late)")
axs2[1].set(xlabel="Total Parameters", ylabel="Filters")
axs2[1].legend()

plt.show()