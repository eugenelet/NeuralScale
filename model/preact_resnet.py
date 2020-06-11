'''Pre-activation ResNet in PyTorch.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''

# based on https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.gate_layer import GateLayer
import numpy as np
import pickle
from utils import compute_params, compute_params_


def norm2d(planes, num_groups=32):
    return nn.BatchNorm2d(planes)


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, group_norm=0, search=False, convcut=False):
        super(PreActBlock, self).__init__()
        self.search = search
        self.bn1 = norm2d(in_planes, group_norm)
        if search:
            self.gate1 = GateLayer(in_planes,in_planes,[1, -1, 1, 1], is_conv=True, is_shortcut=True)

        self.conv1 = nn.Conv2d(in_planes, planes[0], kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm2d(planes[0], group_norm)
        if search:
            self.gate2 = GateLayer(planes[0],planes[0],[1, -1, 1, 1], is_conv=True)

        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)

        if convcut:
            # identity -> conv for all shortcut connection (experimentation purpose)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes[1], kernel_size=1, stride=stride, bias=False)
            )
        else:
            # normal ResNet
            if stride != 1 or in_planes != self.expansion*planes[1]:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes[1], kernel_size=1, stride=stride, bias=False)
                )

    def forward(self, x):
        out = self.bn1(x)
        if self.search:
            out = self.gate1(out)
        out = F.relu(out)

        if hasattr(self, 'shortcut'):
            shortcut = self.shortcut(out)
        else:
            shortcut = x

        out = self.conv1(out)
        out = self.bn2(out)
        if self.search:
            out = self.gate2(out)

        out = F.relu(out)
        out = self.conv2(out)

        out = out + shortcut
        ##as a block here we might benefit with gate at this stage

        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, filters, num_classes=10, group_norm=0, dataset="CIFAR10", search=False, convcut=False):
        super(PreActResNet, self).__init__()

        self.in_planes = filters[0][0]
        self.dataset = dataset
        self.search = search

        self.conv1 = nn.Conv2d(3, filters[0][0], kernel_size=3, stride=1, padding=1, bias=False)

        if num_classes == 200: # tinyimagenet
            self.layer1 = self._make_layer(block, filters[1:1+num_blocks[0]], num_blocks[0], stride=2, group_norm=group_norm, search=search, convcut=convcut)
        else:
            self.layer1 = self._make_layer(block, filters[1:1+num_blocks[0]], num_blocks[0], stride=1, group_norm=group_norm, search=search, convcut=convcut)
        self.layer2 = self._make_layer(block, filters[1+num_blocks[0]:1+sum(num_blocks[:2])], num_blocks[1], stride=2, group_norm=group_norm, search=search, convcut=convcut)
        self.layer3 = self._make_layer(block, filters[1+sum(num_blocks[:2]):1+sum(num_blocks[:3])], num_blocks[2], stride=2, group_norm=group_norm, search=search, convcut=convcut)
        self.layer4 = self._make_layer(block, filters[1+sum(num_blocks[:3]):], num_blocks[3], stride=2, group_norm=group_norm, search=search, convcut=convcut)
        self.bn_out = nn.BatchNorm2d(filters[-1][-1])
        if search:
            self.gate_out = GateLayer(filters[-1][-1], filters[-1][-1], [1, -1, 1, 1], is_conv=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(filters[-1][-1]*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm2d):
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride, group_norm = 0, search=False, convcut=False):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for idx, stride in enumerate(strides):
            layers.append(block(self.in_planes, planes[idx], stride, group_norm = group_norm, search=search, convcut=convcut))
            self.in_planes = planes[idx][-1] * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.bn_out(out)
        out = self.avgpool(out)
        if self.search:
            out = self.gate_out(out)

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def prepare_filters(self, filters, ratio=1, neuralscale=False, iteration=None, search=False, descent_idx=14, prune_fname=None, num_classes=100, pruned_filters=None):
        if ratio!=None:
            if neuralscale and iteration!=0:    
                if search: # perform iterative search of params (architecture desecent)
                    pkl_ld = pickle.load( open( "prune_record/param{}.pk".format(iteration-1), "rb" ) )
                    alpha = pkl_ld["train_alpha"]
                    beta = pkl_ld["train_beta"]
                else: # list of params (done iterative searching/architecture descent)
                    pkl_ld = pickle.load( open( "prune_record/" + prune_fname +".pk", "rb" ) )["param"]
                    alpha = pkl_ld[descent_idx][0]
                    beta = pkl_ld[descent_idx][1]
                total_param = compute_params([cfg*ratio for cfg in sum(filters,[])], classes=num_classes, model='resnet18')
                if np.sum(beta) == 0: 
                    cfg_tmp = list(alpha)
                    ratio_ = 1.2
                    cur_param = 0
                    while abs(cur_param - total_param) > 0.1 * total_param:
                        cur_param = compute_params([int(cfg*ratio_) for cfg in cfg_tmp], classes=num_classes, model='resnet18')
                        if cur_param < total_param:
                            ratio_ += 0.05
                        else:
                            ratio_ -= 0.05
                    filt_cnt = 0
                    new_config = []
                    for block in filters:
                        block_cfg = []
                        for blk_sz in range(len(block)):
                            filt = int(cfg_tmp[filt_cnt]*ratio_)
                            if filt < 10: # filter count too low, add some stochasticity
                                block_cfg.append(filt + 10)
                            else:
                                block_cfg.append(filt)
                            filt_cnt += 1
                        new_config.append(block_cfg)
                else:
                    tau = total_param # initialize tau
                    approx_total_param = 0
                    if descent_idx == 0: # ad-hoc, doesn't converge if too small for idx=0
                        precision = 0.009
                    else:
                        precision = 0.007
                    while abs(approx_total_param - total_param) > int(precision*total_param):
                        approx_filts = []
                        for a,b in zip(alpha,beta):
                            approx_filts.append(int(a*tau**b))
                        approx_total_param = compute_params(approx_filts, classes=num_classes, model='resnet18')
                        tau_update = 0
                        for a,b in zip(alpha,beta):
                            tau_update += a*tau**b * b / tau
                        tau = tau - 1.0 * ((approx_total_param - total_param) * tau_update)
                    filt_cnt = 0
                    new_config = []
                    for idx, block in enumerate(filters):
                        block_cfg = []
                        for blk_sz in range(len(block)):
                            new_filt = int(alpha[filt_cnt]*tau**beta[filt_cnt])
                            if search and new_filt<10: # add stochasticity during training
                                block_cfg.append(new_filt + 10)
                            else:
                                block_cfg.append(new_filt)
                            filt_cnt += 1
                        new_config.append(block_cfg)
                    print(new_config, "approx parameters: {} total parameters: {}".format(approx_total_param, total_param))
            elif pruned_filters != None:
                total_param = compute_params([cfg*ratio for cfg in sum(filters,[])], classes=num_classes, model='resnet18')
                cfg_tmp = pruned_filters
                ratio_ = 1.2
                cur_param = 0
                while abs(cur_param - total_param) > 0.005 * total_param:
                    cur_param = compute_params([int(cfg*ratio_) for cfg in cfg_tmp], classes=num_classes, model='resnet18')
                    if cur_param < total_param:
                        ratio_ += 0.00005
                    else:
                        ratio_ -= 0.00005
                filt_cnt = 0
                new_config = []
                for block in filters:
                    block_cfg = []
                    for blk_sz in range(len(block)):
                        filt = int(cfg_tmp[filt_cnt]*ratio_)
                        block_cfg.append(filt)
                        filt_cnt += 1
                    new_config.append(block_cfg)
                print("pruned uniform", new_config, "cur_params:", cur_param, "total_params:", total_param)
            else: # uniform scale
                new_config = []
                for idx, block in enumerate(filters):
                    block_cfg = []
                    for blk_sz in range(len(block)):
                        block_cfg.append(int(block[blk_sz]*ratio))
                    new_config.append(block_cfg)
                print(new_config)

        else:
            new_config = filters
        # new_config[-1].append(512)
        return new_config

    
    def convert_filters(self, template, filter_list):
        new_config = []
        filt_cnt = 0
        for idx, block in enumerate(template):
            block_cfg = []
            for blk_sz in range(len(block)):
                block_cfg.append(filter_list[filt_cnt])
                filt_cnt += 1
            new_config.append(block_cfg)
        return new_config

def PreActResNet18(filters=None, neuralscale=False, search=False, ratio=None, group_norm = 0, dataset="CIFAR10", iteration=None, num_classes=10, prune_fname="model", descent_idx=0, pruned_filters=None, convcut=False):
    if filters==None:
        filters = [[64],[64,64],[64,64],[128,128],[128,128],[256,256],[256,256],[512,512],[512,512]]
    new_config = PreActResNet.prepare_filters(PreActResNet, filters, ratio, neuralscale, iteration, search, descent_idx, prune_fname, num_classes, pruned_filters)
    model = PreActResNet(PreActBlock, [2,2,2,2], new_config, group_norm= group_norm, dataset=dataset, num_classes=num_classes, search=search, convcut=convcut)
    return model


def test():
    net = PreActResNet18()
    y = net((torch.randn(1,3,32,32)))
    print(y.size())

