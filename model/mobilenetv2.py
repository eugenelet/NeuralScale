"""mobilenetv2 in pytorch

[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen

    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381
"""
# based on https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/mobilenetv2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.gate_layer import GateLayer
import pickle
import numpy as np
from utils import compute_params, compute_params_


class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6, num_classes=100, search=False, convcut=False):
        super().__init__()
        self.search = search
        if search:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, in_channels * t, 1),
                nn.BatchNorm2d(in_channels * t),
                nn.ReLU6(inplace=True),

                nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
                nn.BatchNorm2d(in_channels * t),
                nn.ReLU6(inplace=True),

                nn.Conv2d(in_channels * t, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
            self.gate_residual = GateLayer(out_channels, out_channels, [1, -1, 1, 1], is_conv=True)
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, in_channels * t, 1),
                nn.BatchNorm2d(in_channels * t),
                nn.ReLU6(inplace=True),

                nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
                nn.BatchNorm2d(in_channels * t),
                nn.ReLU6(inplace=True),

                nn.Conv2d(in_channels * t, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )

        if convcut:
            # Identiy -> Conv (for experimentation purpose)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(out_channels),
            )
        else:
            # Normal MobileNetV2
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                    nn.BatchNorm2d(out_channels),
                )

    
    def forward(self, x):

        residual = self.residual(x)

        if hasattr(self, 'shortcut'):
            residual += self.shortcut(x)
        else:
            residual += x
        if self.search:
            residual = self.gate_residual(residual)
        
        return residual

class MobileNetV2(nn.Module):

    def __init__(self, num_classes=100, filters=None,
                 neuralscale=False, search=False, ratio=None,
                 dataset="CIFAR10", iteration=None,
                 prune_fname="model", descent_idx=0, pruned_filters=None, convcut=False):
        super().__init__()

        if filters==None:
            filters = [[32],[16],[24,24],[32,32,32],[64,64,64,64],[96,96,96],[160,160,160],[320],[1280]]

        filters = self.prepare_filters(filters, ratio, neuralscale, iteration, search, descent_idx, prune_fname, num_classes, pruned_filters)
        self.search = search
        if search:
            self.pre = nn.Sequential(
                nn.Conv2d(3, filters[0][0], 3, padding=1),
                nn.BatchNorm2d(filters[0][0]),
            )
            self.gate_pre = GateLayer(filters[0][0], filters[0][0], [1, -1, 1, 1], is_conv=True)
        else:
            self.pre = nn.Sequential(
                nn.Conv2d(3, filters[0][0], 3, padding=1),
                nn.BatchNorm2d(filters[0][0]),
            )


        self.stage1 = LinearBottleNeck(filters[0][-1], filters[1][0], 1, 1, search=search, convcut=convcut)
        #                            repeat     in            out     stride   t
        self.stage2 = self._make_stage(2, filters[1][-1], filters[2],   2,     6, search=search, convcut=convcut)
        self.stage3 = self._make_stage(3, filters[2][-1], filters[3],   2,     6, search=search, convcut=convcut)
        self.stage4 = self._make_stage(4, filters[3][-1], filters[4],   2,     6, search=search, convcut=convcut)
        if num_classes == 200: # tinyimagenet
            self.stage5 = self._make_stage(3, filters[4][-1], filters[5],   2,     6, search=search, convcut=convcut)
        else:
            self.stage5 = self._make_stage(3, filters[4][-1], filters[5],   1,     6, search=search, convcut=convcut)
        self.stage6 = self._make_stage(3, filters[5][-1], filters[6],   2,     6, search=search, convcut=convcut)
        self.stage7 = LinearBottleNeck(filters[6][-1], filters[7][0], 1, 6, search=search, convcut=convcut)

        self.conv1 = nn.Sequential(
            nn.Conv2d(filters[7][0], filters[8][0], 1),
            nn.BatchNorm2d(filters[8][0]),
        )
        if search:
            self.gate_conv1 = GateLayer(filters[8][0], filters[8][0], [1, -1, 1, 1], is_conv=True)
        

        self.conv2 = nn.Conv2d(filters[8][0], num_classes, 1)

    def forward(self, x):
        x = self.pre(x)
        if self.search:
            x = self.gate_pre(x)
        x = F.relu6(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        if self.search:
            x = self.gate_conv1(x)
        x = F.relu6(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        return x

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
                total_param = compute_params([int(cfg*ratio) for cfg in sum(filters,[])], classes=num_classes, model='mobilenetv2', last=True)
                # fallback to uniform scaling
                if np.sum(beta) == 0: 
                    cfg_tmp = list(alpha)
                    ratio_ = 1.2
                    cur_param = 0
                    while abs(cur_param - total_param) > 0.1 * total_param:
                        cur_param = compute_params([int(cfg*ratio_) for cfg in cfg_tmp], classes=num_classes, model='mobilenetv2', last=True)
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
                    filters=new_config
                else:
                    tau = total_param # initialize tau
                    approx_total_param = 0
                    while abs(approx_total_param - total_param) > int(0.0005*total_param):
                        approx_filts = []
                        for a,b in zip(alpha,beta):
                            approx_filts.append(a*tau**b)
                        approx_total_param = compute_params(approx_filts, classes=num_classes, model='mobilenetv2', last=True)
                        tau_update = 0
                        for a,b in zip(alpha,beta):
                            tau_update += a*tau**b * b / tau
                        if ((approx_total_param - total_param) * tau_update) > tau:
                            tau *= 0.5
                        else:
                            tau = tau - 1.0 * ((approx_total_param - total_param) * tau_update)
                    filt_cnt = 0
                    new_config = []
                    for idx, block in enumerate(filters):
                        block_cfg = []
                        for blk_sz in range(len(block)):
                            filt = int(alpha[filt_cnt]*tau**beta[filt_cnt])
                            if search: # only add stochasticity during architecture searching
                                if filt < 10: # filter count too low, add some stochasticity
                                    block_cfg.append(filt + 10)
                                else:
                                    block_cfg.append(filt)
                            else:
                                block_cfg.append(filt)
                            filt_cnt += 1
                        new_config.append(block_cfg)
                    print(new_config, "approx parameters: {} total parameters: {}".format(approx_total_param, total_param))
                    filters = new_config
            elif pruned_filters != None:
                total_param = compute_params([int(cfg*ratio) for cfg in sum(filters,[])], classes=num_classes, model='mobilenetv2', last=True)
                cfg_tmp = pruned_filters
                ratio_ = 1.2
                cur_param = 0
                while abs(cur_param - total_param) > 0.005 * total_param:
                    cur_param = compute_params([int(cfg*ratio_) for cfg in cfg_tmp], classes=num_classes, model='mobilenetv2', last=True)
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
                filters = new_config
                print("pruned uniform", new_config, "cur_params:", cur_param, "total_params:", total_param)
            else: # uniform scale
                new_config = []
                for idx, block in enumerate(filters):
                    block_cfg = []
                    for blk_sz in range(len(block)):
                        block_cfg.append(int(block[blk_sz]*ratio))
                    new_config.append(block_cfg)
                filters = new_config
                print(filters)
        return filters

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

    
    def _make_stage(self, repeat, in_channels, out_channels, stride, t, search=False, convcut=False):
        layers = []
        for out_ch in out_channels:
            layers.append(LinearBottleNeck(in_channels, out_ch, stride, t, search=search, convcut=convcut))
            in_channels = out_ch
            stride = 1
        return nn.Sequential(*layers)


if __name__ == "__main__":
    model = MobileNetV2(ratio=1, neuralscale=True, iteration=1, num_classes=100, search=True, dataset="CIFAR100")
