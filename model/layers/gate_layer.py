"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""


import torch
import torch.nn as nn

'''
Gating layer for pruning
'''

class GateLayer(nn.Module):
    def __init__(self, input_features, output_features, size_mask, kernel_size=3, is_conv=False, is_shortcut=False):
        super(GateLayer, self).__init__()
        # FLOPs computation
        self.input_features = input_features
        self.output_features = output_features
        self.is_conv = is_conv # indicator if its a gate for conv layers
        self.is_shortcut = is_shortcut # for ResNet only
        self.feat_size = 0 # initialized using setup_flops(model, train_loader)

        self.size_mask = size_mask
        self.weight = nn.Parameter(torch.ones(output_features))
        self.kernel_size = kernel_size
        # for simpler way to find these layers
        self.do_not_update = True

        # indicator that pruning has begun for this layer
        self.begin_prune = False

    def forward(self, input):
        return input*self.weight.view(*self.size_mask)

    def extra_repr(self):
        return 'in_features={}, out_features={}'.format(
            self.input_features, self.output_features is not None
        )