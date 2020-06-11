'''
Fits pattern formed by residual fitlers of pruning using least-squares as proposed in paper
'''

import csv
import numpy as np
import pickle
from cvxopt import matrix, solvers
from utils import compute_params
import argparse

solvers.options['show_progress'] = False

def fit_params(iteration=None, prune_fname="filename", classes=10, model='vgg', in_channel=3, kernel=3):
    if iteration==None:
        f = open('prune_record/train.csv', newline='')
    else:
        f = open('prune_record/' + prune_fname + '_{}.csv'.format(iteration), newline='')
    reader = csv.reader(f, delimiter=',')
    filters = []
    for row in reader:
        filters.append(list(map(int,row)))

    filters = np.array(filters, dtype=int)
    # Samples insuffcient to get good interpolation
    if filters.shape[0] < 6:
        filters = np.expand_dims(filters[0],axis=0)
    if filters.shape[0] == 1: # not all layers pruned at least once, opt for uniform scaling
        alpha = filters[0]
        beta = np.zeros(filters.shape[1])
        # =======================
        # save scaling parameters
        # =======================
        pickle_save = {
            "train_alpha": alpha,
            "train_beta": beta,
        }
        if iteration != None:
            pickle_out = open("prune_record/param{}.pk".format(iteration),"wb")
        else:
            pickle_out = open("prune_record/param.pk","wb")
        pickle.dump(pickle_save, pickle_out)
        pickle_out.close()

        return alpha, beta
    # Compute total parameters
    total_params = []
    for filt in filters: # over all iterations
        total_params.append(compute_params(filt, classes=classes, model=model, in_channel=in_channel, kernel=kernel, last=True))
    total_params = np.array(total_params)

    # ######
    # LR (simple)
    # ######
    ln_tau = np.log(total_params)
    Tau = np.stack((ln_tau, np.ones(ln_tau.shape)), axis=1) 
    Phi = np.log(filters)
    Theta = np.matmul(np.matmul(np.linalg.inv( np.matmul(Tau.transpose(), Tau)), Tau.transpose()), Phi)
    beta = Theta[0,:]
    alpha = np.exp(Theta[1,:])

    f.close()

    # =======================
    # save scaling parameters
    # =======================
    pickle_save = {
        "train_alpha": alpha,
        "train_beta": beta,
    }

    if iteration != None:
        pickle_out = open("prune_record/param{}.pk".format(iteration),"wb")
    else:
        pickle_out = open("prune_record/param.pk","wb")
    pickle.dump(pickle_save, pickle_out)
    pickle_out.close()

    return alpha, beta

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Search parameters using residual filters')
    parser.add_argument('--dataset', default="tinyimagenet", type=str,
                            help='dataset for experiment, choice: CIFAR10, CIFAR100, tinyimagenet', choices= ["tinyimagenet", "CIFAR10", "CIFAR100"])
    parser.add_argument('--prune_fname', default='filename',
                        help='prune save file')
    parser.add_argument('--model', default="vgg", type=str,
                        help='model selection, choices: vgg, mobilenetv2, resnet18',
                        choices=["vgg", "mobilenetv2", "resnet18"])
    args = parser.parse_args()

    if args.dataset == "CIFAR10":
        num_classes = 10
    elif args.dataset == "CIFAR100":
        num_classes = 100
    if args.dataset == "tinyimagenet":
        num_classes = 200
    param_list = []
    for iteration in range(15):   
        alpha, beta = fit_params(iteration=iteration, prune_fname=args.prune_fname, classes=num_classes, model=args.model) # search for params of each layer
        param_list.append([alpha,beta])
    
    # =======================
    # save scaling parameters
    # =======================
    pickle_save = {
        "param": param_list,
    }

    pickle_out = open("prune_record/" + args.prune_fname + ".pk","wb")
    pickle.dump(pickle_save, pickle_out)
    pickle_out.close()

