"""
Modified from: https://github.com/NVlabs/Taylor_pruning/blob/master/pruning_engine.py
"""

from __future__ import print_function

import numpy as np
import csv
import torch.nn as nn
import torch

class pruner(object):
    def __init__(self, parameters, iteration=None, prune_fname="filename", flops_on=False, classes=10, model=None, prune_neurons=0.95):
        self.classes = classes
        self.model = model
        # store some statistics
        self.min_criteria_value = 1e6
        self.max_criteria_value = 0.0
        self.median_criteria_value = 0.0
        self.neuron_units = 0
        self.all_neuron_units = 0
        self.pruned_neurons = 0
        self.pruning_iterations_done = 0
        self.prune_per_iteration = 10
        self.starting_neuron = 0
        self.group_size = 1
        if classes==200: #tinyimagenet:
            self.frequency = 100
        else:
            self.frequency = 50

        self.parameters = list()
        self.gate_layers = list()

        ##get pruning parameters
        for parameter in parameters:
            parameter_value = parameter["parameter"]
            self.parameters.append(parameter_value)
            self.gate_layers.append(parameter["layer"])

        ##prune all layers
        self.prune_layers = [True for parameter in self.parameters]

        self.iterations_done = 0

        self.prune_network_criteria = list()
        self.prune_network_accumulate = {"by_layer": list(), "averaged": list(), "averaged_cpu": list()}

        self.pruning_gates = list()
        for layer in range(len(self.parameters)):
            self.prune_network_criteria.append(list())

            for key in self.prune_network_accumulate.keys():
                self.prune_network_accumulate[key].append(list())

            self.pruning_gates.append(np.ones(len(self.parameters[layer]),))
            layer_now_criteria = self.prune_network_criteria[-1]
            for unit in range(len(self.parameters[layer])):
                layer_now_criteria.append(0.0)

    

        # the rest of initializations
        self.pruned_neurons = self.starting_neuron

        # stores results of the pruning, 0 - unsuccessful, 1 - successful
        self.res_pruning = 0

        self.iter_step = -1

        # set number of pruned neurons to be a certain percentage
        all_neuron_units, neuron_units = self._count_number_of_neurons()
        self.prune_neurons_max = int(all_neuron_units * prune_neurons)
        self.maximum_pruning_iterations = int(np.ceil(self.prune_neurons_max / self.prune_per_iteration))

        # record pruned filter dimension into file
        if iteration != None:
            self.f = open("./prune_record/" + prune_fname + "_{}.csv".format(iteration), "w", newline="")
        else:
            self.f = open("./prune_record/train.csv", "w", newline="")
        self.writer = csv.writer(self.f)
        self.recorded_filters = list()

    def add_criteria(self):
        '''
        This method adds criteria to global list given batch stats.
        '''
        for layer, if_prune in enumerate(self.prune_layers):
            # Taylor pruning on gate
            nunits = self.parameters[layer].size(0)
            criteria_for_layer = (self.parameters[layer]*self.parameters[layer].grad).data.pow(2).view(nunits, -1).sum(dim=1)
            if self.iterations_done == 0:
                self.prune_network_accumulate["by_layer"][layer] = criteria_for_layer
            else:
                self.prune_network_accumulate["by_layer"][layer] += criteria_for_layer

        self.iterations_done += 1

    @staticmethod
    def group_criteria(list_criteria_per_layer, group_size=1):
        '''
        Function combine criteria per neuron into groups of size group_size.
        Output is a list of groups organized by layers. Length of output is a number of layers.
        The criterion for the group is computed as an average of member's criteria.
        Input:
        list_criteria_per_layer - list of criteria per neuron organized per layer
        group_size - number of neurons per group

        Output:
        groups - groups organized per layer. Each group element is a tuple of 2: (index of neurons, criterion)
        '''
        groups = list()

        for layer in list_criteria_per_layer:
            layer_groups = list()
            indeces = np.argsort(layer)
            for group_id in range(int(np.ceil(len(layer)/group_size))):
                current_group = slice(group_id*group_size, min((group_id+1)*group_size, len(layer)))
                values = [layer[ind] for ind in indeces[current_group]]
                group = [indeces[current_group], sum(values)]

                layer_groups.append(group)
            groups.append(layer_groups)

        return groups

    def compute_saliency(self):
        '''
        Method performs pruning based on precomputed criteria values. Needs to run after add_criteria()
        '''

        if self.maximum_pruning_iterations <= self.pruning_iterations_done:
            # if reached max number of pruning iterations -> exit
            self.res_pruning = -1
            return -1

        self.full_list_of_criteria = list()

        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue

            if self.iterations_done > 0:
                contribution = self.prune_network_accumulate["by_layer"][layer] / self.iterations_done
                self.prune_network_accumulate["averaged"][layer] = contribution

                current_layer = self.prune_network_accumulate["averaged"][layer]
                current_layer = current_layer.cpu().numpy()

                self.prune_network_accumulate["averaged_cpu"][layer] = current_layer
            else:
                print("First do some add_criteria iterations")
                exit()

            for unit in range(len(self.parameters[layer])):
                criterion_now = current_layer[unit]

                # make sure that pruned neurons have 0 criteria
                self.prune_network_criteria[layer][unit] =  criterion_now * self.pruning_gates[layer][unit]


        # count number of neurons
        all_neuron_units, neuron_units = self._count_number_of_neurons()
        self.neuron_units = neuron_units
        self.all_neuron_units = all_neuron_units

       
        self.iterations_done = 0

        # create groups per layer
        groups = self.group_criteria(self.prune_network_criteria, group_size=self.group_size)

        # get an array of all criteria from groups
        all_criteria = np.asarray([group[1] for layer in groups for group in layer]).reshape(-1)

        prune_neurons_now = (self.pruned_neurons + self.prune_per_iteration)//self.group_size - 1
        if self.prune_neurons_max != -1:
            prune_neurons_now = min(len(all_criteria)-1, min(prune_neurons_now, self.prune_neurons_max//self.group_size - 1))

        # adaptively estimate threshold given a number of neurons to be removed
        threshold_now = np.sort(all_criteria)[prune_neurons_now]


        self.pruning_iterations_done += 1

        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue

            if self.prune_per_iteration == 0:
                continue

            for group in groups[layer]:
                if group[1] <= threshold_now:
                    if sum(self.pruning_gates[layer]) == 1:
                        self.res_pruning = -1
                        return -1 # last surviving neuron, terminate pruning process
                    for unit in group[0]:
                        # do actual pruning
                        self.pruning_gates[layer][unit] *= 0.0
                        self.parameters[layer].data[unit] *= 0.0
                        self.gate_layers[layer].begin_prune = True

        # count number of neurons
        all_neuron_units, neuron_units = self._count_number_of_neurons()

        self.pruned_neurons = all_neuron_units-neuron_units

        self.threshold_now = threshold_now

        try:
            self.min_criteria_value = (all_criteria[all_criteria > 0.0]).min()
            self.max_criteria_value = (all_criteria[all_criteria > 0.0]).max()
            self.median_criteria_value = np.median(all_criteria[all_criteria > 0.0])
        except:
            self.min_criteria_value = 0.0
            self.max_criteria_value = 0.0
            self.median_criteria_value = 0.0

        # set result to successful
        self.res_pruning = 1
        return 1

    def _count_number_of_neurons(self):
        '''
        Function computes number of total neurons and number of active neurons
        :return:
        all_neuron_units - number of neurons considered for pruning
        neuron_units     - number of not pruned neurons in the model
        '''
        all_neuron_units = 0
        neuron_units = 0
        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue

            all_neuron_units += len( self.parameters[layer] )
            neuron_units += int(torch.norm(self.parameters[layer],1).data.cpu().item())

        return all_neuron_units, neuron_units

    def report_filter_number(self, save=False, force_save=False):
        '''
        Function reports the remaining number of filters/neurons of each layer and stores them
        '''
        filters = []
        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue

            # neuron_units = 0
            neuron_units = int(torch.norm(self.parameters[layer],1).data.cpu().item())
            filters.append(neuron_units)

        print(filters)
        if save==True and self.check_begin_prune()==True:
            self.recorded_filters.append(filters)
        if force_save==True:
            self.recorded_filters.append(filters)

    def compute_flops(self, im_ch=3, classes=10, as_reg=False):
        '''
        Function computes the flops of each layer
        :return:
        flops - flops contribution of individual layers
        total_flops - flops of entire network
        '''
        total_flops = 0
        flops = []
        in_neurons = im_ch # initialize input channel of first layer
        shortcut_in = 0 # skip first encounter of shortcut
        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue

            if as_reg:
                neuron_units = torch.norm(self.parameters[layer],1)
            else:
                neuron_units = torch.norm(self.parameters[layer],1).data.cpu().item()
            if self.gate_layers[layer].is_conv == True: # conv layer
                lyr_flops = 2*neuron_units*in_neurons*self.gate_layers[layer].kernel_size*self.gate_layers[layer].kernel_size*self.gate_layers[layer].feat_size*self.gate_layers[layer].feat_size
                if self.gate_layers[layer].is_shortcut: # residual layers (ResNet only)
                    lyr_flops += 2*neuron_units*shortcut_in*self.gate_layers[layer].feat_size*self.gate_layers[layer].feat_size # 1-by-1 kernel
                    shortcut_in = neuron_units # input of shortcut layer
            else: # linear layer
                lyr_flops = 2*neuron_units*in_neurons
            flops.append(lyr_flops)
            total_flops += lyr_flops
            in_neurons = neuron_units # update input channel/neurons
        # Output layer
        if self.model[:3] == "res": # resnet
            flops.append(2*neuron_units*classes)
            total_flops += 2*neuron_units*classes
        elif self.model[:3] == "vgg": # VGG
            flops.append(2*neuron_units*classes/4)
            total_flops += 2*neuron_units*classes/4
        else:
            print("Model error!")
            exit()
        return flops, total_flops

    
    def check_begin_prune(self):
        '''
        Check if all layers are pruned by at least 1 neurons (to begin recording of prune layers)
        '''
        for layer in self.gate_layers:
            if layer.begin_prune == False:
                return False
        return True
        
    def enforce_pruning(self):
        '''
        Method sets parameters ang gates to 0 for pruned neurons.
        Helpful if optimizer will change weights from being zero (due to regularization etc.)
        '''
        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue

            for unit in range(len(self.parameters[layer])):
                if self.pruning_gates[layer][unit] == 0.0:
                    self.parameters[layer].data[unit] *= 0.0
                    self.gate_layers[layer].begin_prune = True


    # def util_add_loss(self, training_loss_current, training_acc):
    #     # keeps track of current loss
    #     self.util_loss_tracker += training_loss_current
    #     self.util_acc_tracker  += training_acc
    #     self.util_loss_tracker_num += 1

    def do_step(self, loss=None, optimizer=None, neurons_left=0, training_acc=0.0):
        '''
        do one step of pruning,
        1) Add importance estimate
        2) checks if loss is above threshold
        3) performs one step of pruning if needed
        '''
        self.iter_step += 1
        niter = self.iter_step

        # stop if pruned maximum amount
        if self.maximum_pruning_iterations <= self.pruning_iterations_done:
            # exit if we pruned enough
            self.res_pruning = -1
            return -1

        # compute criteria for given batch
        self.add_criteria()

        if niter % self.frequency == 0 and niter != 0:
            # do actual pruning, output: 1 - good, 0 - no pruning

            self.compute_saliency()
            self.set_momentum_zero_sgd(optimizer=optimizer)

            # training_loss = self.util_training_loss
            if self.res_pruning == 1:
                print("Pruning: Units", self.neuron_units, "/", self.all_neuron_units, "Zeroed", self.pruned_neurons, "criteria min:{}/max:{:2.7f}".format(self.min_criteria_value,self.max_criteria_value))
                self.report_filter_number(save=True)
                return 1 # neurons pruned sucessfully
            else:
                return -1

    def set_momentum_zero_sgd(self, optimizer=None):
        '''
        Method sets momentum buffer to zero for pruned neurons. Supports SGD only.
        :return:
        void
        '''
        for layer in range(len(self.pruning_gates)):
            if not self.prune_layers[layer]:
                continue
            for unit in range(len(self.pruning_gates[layer])):
                if self.pruning_gates[layer][unit]:
                    continue
                if 'momentum_buffer' in optimizer.state[self.parameters[layer]].keys():
                    optimizer.state[self.parameters[layer]]['momentum_buffer'][unit] *= 0.0

def prepare_pruning_list(model, unstructured=False):
    '''
    Function returns a list of parameters from model to be considered for pruning.
    Depending on the pruning method and strategy different parameters are selected (conv kernels, BN parameters etc)
    :param pruning_settings:
    :param model:
    :return:
    '''
    # Function creates a list of layer that will be pruned based o user selection

    pruning_parameters_list = list()

    print("network structure")
    for module_indx, m in enumerate(model.modules()):
        # print(module_indx, m)
        if unstructured:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m_to_add = m
                for_pruning = {"parameter": m_to_add.weight, "layer": m_to_add,
                                "compute_criteria_from": m_to_add.weight}
                pruning_parameters_list.append(for_pruning)
        else:
            if hasattr(m, "do_not_update"):
                m_to_add = m
                for_pruning = {"parameter": m_to_add.weight, "layer": m_to_add,
                                "compute_criteria_from": m_to_add.weight}
                pruning_parameters_list.append(for_pruning)

    return pruning_parameters_list


feats = []
def hook(module, input, output):
    feats.append(output)

def setup_flops(model, train_loader, use_cuda=True):
    '''
    Function returns a list of parameters from model to be considered for pruning.
    Depending on the pruning method and strategy different parameters are selected (conv kernels, BN parameters etc)
    :param pruning_settings:
    :param model:
    :return:
    '''
    global feats
    hooks = []
    for m in model.modules():
        if hasattr(m, "do_not_update"):
            hooks.append(m.register_forward_hook(hook))
    model.train()
    for (data, target) in train_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # make sure that all gradients are zero
        for p in model.parameters():
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()
        model(data)
        break
    
    hook_cnt = 0
    for m in model.modules():
        if hasattr(m, "do_not_update"): # gate layer
            if len(feats[hook_cnt]==4): # output of conv layer
                m.feat_size = feats[hook_cnt].shape[3] # output feature map size
                hooks[hook_cnt].remove()
            hook_cnt += 1
    hooks = []
    feats = []



if __name__ == '__main__':
    pass
