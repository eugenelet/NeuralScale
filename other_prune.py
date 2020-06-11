'''
Compares other pruning method for experimentation prupose
METHOD
0: Liu, Zhuang, et al. "Learning efficient convolutional networks through network slimming." Proceedings of the IEEE International Conference on Computer Vision. 2017.
1: Li, Hao, et al. "Pruning filters for efficient convnets." arXiv preprint arXiv:1608.08710 (2016).
'''

import numpy as np
import csv
import torch.nn as nn
import torch
from utils import compute_params

class compare_pruner(object):
    def __init__(self, model, method=0, size=0.25, model_name="vgg", dataset="CIFAR10"):
        self.method = method
        self.size = size
        self.model_name = model_name
        self.dataset = dataset
        self.prune_per_iteration = 10

        prev_module = None
        prev_module2 = None
        pruning_parameters_list = list()
        for module_indx, m in enumerate(model.modules()):
            if hasattr(m, "do_not_update"):
                if method == 0: # slimming
                    for_pruning = {"parameter": m.weight, "layer": m,
                                    "compute_criteria_from": [prev_module.weight, prev_module.bias]}
                elif method == 1: # weight norm
                    for_pruning = {"parameter": m.weight, "layer": m,
                                    "compute_criteria_from": prev_module2.weight}
                else:
                    print("No such method.")
                    exit()
                pruning_parameters_list.append(for_pruning)
            if isinstance(prev_module, nn.Conv2d):
                prev_module2 = prev_module
            prev_module = m

        
        self.parameters = list()
        self.gate_layers = list()
        self.prune_criteria = list()

        ##get pruning parameters
        for parameter in pruning_parameters_list:
            self.parameters.append(parameter["parameter"])
            self.gate_layers.append(parameter["layer"])
            self.prune_criteria.append(parameter["compute_criteria_from"])

        self.prune_layers = [True for parameter in pruning_parameters_list]
        self.pruned_neurons = 0

        self.pruning_gates = list()
        for layer in range(len(self.parameters)):
            self.pruning_gates.append(np.ones(len(self.parameters[layer])))

    def prune_neurons(self, optimizer):
        if self.dataset == "CIFAR10":
            num_classes = 10
        elif self.dataset == "CIFAR100":
            num_classes = 100
        elif self.dataset == "tinyimagenet":
            num_classes = 200

        # set number of pruned neurons to be a certain percentage
        all_neuron_units, neuron_units = self._count_number_of_neurons()
        filters = self.compute_filter_number()
        targeted_filter = [filt*self.size for filt in filters]
        targeted_params = compute_params(targeted_filter, classes=num_classes, model=self.model_name)
        cur_params = compute_params(filters, classes=num_classes, model=self.model_name)
        print("Before: ", filters)
        ratio = 0.9
        # while abs(cur_params - targeted_params) > int(targeted_params*0.0005):
        while targeted_params < cur_params:
            if self.method == 0: # network slimming
                all_criteria = torch.tensor([abs(criterion) for layer_criteria in self.prune_criteria for criterion in layer_criteria[0]]).cuda()
                prune_neurons_now = self.pruned_neurons + self.prune_per_iteration
                threshold_now = torch.sort(all_criteria)[0][prune_neurons_now]
                for layer, layer_criteria in enumerate(self.prune_criteria):
                    for unit, criterion in enumerate(layer_criteria[0]):
                        if abs(criterion) <= threshold_now:
                            # do actual pruning
                            self.pruning_gates[layer][unit] *= 0.0
                            self.parameters[layer].data[unit] *= 0.0
                            self.prune_criteria[layer][0].data[unit] *= 0.0 # weight
                            self.prune_criteria[layer][1].data[unit] *= 0.0 # bias (not important)

                # count number of neurons
                all_neuron_units, neuron_units = self._count_number_of_neurons()
                self.pruned_neurons = all_neuron_units-neuron_units
                cur_filter = self.compute_filter_number()
                cur_params = compute_params(cur_filter, classes=num_classes, model=self.model_name)
                print(cur_params, cur_filter)
            elif self.method == 1: # uniformly pruned across all layers
                cur_filter = [int(filt*ratio) for filt in filters]
                cur_params = compute_params(cur_filter, classes=num_classes, model=self.model_name)
                ratio *= 0.999
            else:
                print("No such method")
                exit()

        if self.method == 1: # weight magnitude
            for layer, target_filt in enumerate(cur_filter):
                layer_criteria = np.asarray([torch.norm(filt,1).data.cpu().item() for filt in self.prune_criteria[layer]]).reshape(-1)

                # adaptively estimate threshold given a number of neurons to be removed
                threshold_now = np.sort(layer_criteria)[::-1][target_filt]

                for unit, criterion in enumerate(layer_criteria):
                    if abs(criterion) <= threshold_now:
                        # do actual pruning
                        self.pruning_gates[layer][unit] *= 0.0
                        self.parameters[layer].data[unit] *= 0.0
                        self.prune_criteria[layer].data[unit,:] *= 0.0
            cur_filter = self.compute_filter_number()


        # Set momentum buffer to 0
        for layer in range(len(self.pruning_gates)):
            for unit in range(len(self.pruning_gates[layer])):
                if self.pruning_gates[layer][unit]:
                    continue
                if 'momentum_buffer' in optimizer.state[self.parameters[layer]].keys():
                    optimizer.state[self.parameters[layer]]['momentum_buffer'][unit] *= 0.0
        print("After: ", cur_filter)
        print("Target Params:", targeted_params, " Approx. Param:", cur_params)


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
    
    def compute_filter_number(self):
        '''
        Function reports the remaining number of filters/neurons of each layer and stores them
        '''
        filters = []
        for layer, if_prune in enumerate(self.prune_layers):
            if not if_prune:
                continue

            neuron_units = int(torch.norm(self.parameters[layer],1).data.cpu().item())
            filters.append(neuron_units)

        return filters
