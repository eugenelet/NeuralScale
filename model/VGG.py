"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

# based on https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from model.layers.gate_layer import GateLayer
import pickle
import numpy as np
from utils import compute_params, compute_params_

__all__ = [
    'vgg11',
]

model_urls = {
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
}

class LinView(nn.Module):
    def __init__(self):
        super(LinView, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class VGG(nn.Module):
    def __init__(self, features, config, num_classes=10, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Linear(config[-2], num_classes)
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out')#, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def prepare_filters(self, config, ratio=1, neuralscale=False, iteration=None, search=False, descent_idx=14, prune_fname=None, num_classes=100, pruned_filters=None):
        if ratio != None: # for ratio swiping
            if neuralscale and iteration!=0: # use proposed efficient scaling method
                if search: # perform iterative search of params (architecture desecent)
                    pkl_ld = pickle.load( open( "prune_record/param{}.pk".format(iteration-1), "rb" ) )
                    alpha = pkl_ld["train_alpha"]
                    beta = pkl_ld["train_beta"]
                else: # list of params (done iterative searching/architecture descent)
                    pkl_ld = pickle.load( open( "prune_record/" + prune_fname +".pk", "rb" ) )["param"]
                    alpha = pkl_ld[descent_idx][0]
                    beta = pkl_ld[descent_idx][1]

                # total_param = compute_params_(vgg11(config=self.convert_filters(self, template=config, filter_list=[int(cfg*ratio) for cfg in list(filter(lambda a: a != 'M', config))]), num_classes=num_classes) )
                total_param = compute_params([cfg*ratio for cfg in list(filter(lambda a: a != 'M', config))], classes=num_classes, model='vgg')
                tau = total_param # initialize tau
                for j in range(2000):
                    approx_filts = []
                    for a,b in zip(alpha,beta):
                        approx_filts.append(int(a*tau**b))
                    # approx_total_param = compute_params(vgg11(config=self.convert_filters(self, template=config, filter_list=approx_filts), num_classes=num_classes) )
                    approx_total_param = compute_params(approx_filts, classes=num_classes, model='vgg')
                    tau_update = 0
                    for a,b in zip(alpha,beta):
                        tau_update += a*tau**b * b / tau 
                    tau = tau - 50.0 * ((approx_total_param - total_param) * tau_update)
                new_config = []
                cfg_cnt = 0
                for i in range(len(config)):
                    if config[i] != 'M':
                        new_config.append(int(alpha[cfg_cnt]*tau**beta[cfg_cnt]))
                        cfg_cnt += 1
                    else:
                        new_config.append(config[i]) # M
                print(new_config, "approx params: {} total params: {}".format(approx_total_param, total_param))
            elif pruned_filters != None:
                total_param = compute_params([cfg*ratio for cfg in list(filter(lambda a: a != 'M', config))], classes=num_classes, model='vgg')
                cfg_tmp = pruned_filters
                ratio_ = 1.2
                cur_param = 0
                while abs(cur_param - total_param) > 0.005 * total_param:
                    cur_param = compute_params([int(cfg*ratio_) for cfg in cfg_tmp], classes=num_classes, model='vgg')
                    if cur_param < total_param:
                        ratio_ += 0.00005
                    else:
                        ratio_ -= 0.00005
                filt_cnt = 0
                new_config = []
                for i in range(len(config)):
                    if config[i] != 'M':
                        new_config.append(int(cfg_tmp[filt_cnt]*ratio_))
                        filt_cnt += 1
                    else:
                        new_config.append(config[i]) # M
                print("pruned uniform", new_config, "cur_params:", cur_param, "total_params:", total_param)
            else: # uniform scaling
                new_config = []
                for i in range(len(config)):
                    if config[i] != 'M':
                        new_config.append(int(config[i] * ratio))
                    else:
                        new_config.append(config[i]) # M
        else:
            new_config = config
        return new_config
    

    def convert_filters(self, template, filter_list):
        new_config = []
        cfg_cnt = 0
        for i in range(len(template)):
            if template[i] != 'M':
                new_config.append(filter_list[cfg_cnt])
                cfg_cnt += 1
            else:
                new_config.append(template[i]) # M
        return new_config

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}


def flatten_model(old_net):
    """Removes nested modules. Only works for VGG."""
    from collections import OrderedDict
    module_list, counter, inserted_view = [], 0, False
    gate_counter = 0
    # print("printing network")
    # print(" Hard codded network in vgg_bn.py")
    for m_indx, module in enumerate(old_net.modules()):
        if not isinstance(module, (nn.Sequential, VGG)):
            # print(m_indx, module)
            if isinstance(module, nn.Linear) and not inserted_view:
                module_list.append(('flatten', LinView()))
                inserted_view = True

            # features.0
            # classifier
            prefix = "features"

            if m_indx > 30:
                prefix = "classifier"
            if m_indx == 32:
                counter = 0

            # prefix = ""

            module_list.append((prefix + str(counter), module))

            if isinstance(module, nn.BatchNorm2d):
                planes = module.num_features
                gate = GateLayer(planes, planes, [1, -1, 1, 1], is_conv=True)
                module_list.append(('gate%d' % (gate_counter), gate))
                # print("gate ", counter, planes)
                gate_counter += 1


            if isinstance(module, nn.BatchNorm1d):
                planes = module.num_features
                gate = GateLayer(planes, planes, [1, -1], is_conv=False)
                module_list.append(('gate%d' % (gate_counter), gate))
                # print("gate ", counter, planes)
                gate_counter += 1


            counter += 1
    new_net = nn.Sequential(OrderedDict(module_list))
    return new_net


def vgg11(pretrained=False, config=None, ratio=None, neuralscale=False, iteration=None, search=False, num_classes=10, prune_fname="model", descent_idx=0, pruned_filters=None, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    if config == None:
        config = cfg['A']
    new_config = VGG.prepare_filters(VGG, config, ratio, neuralscale, iteration, search, descent_idx, prune_fname, num_classes, pruned_filters)
    model = VGG(make_layers(new_config, batch_norm=True), new_config, num_classes=num_classes, **kwargs)
    model = flatten_model(model)
    return model
