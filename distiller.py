import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import scipy
import numpy as np
import math


def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8

def similarity(feat):
    feat = feat.float()
    tmp = L2(feat).detach()
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
    return torch.einsum('icm,icn->imn', [feat, feat])

def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis

def distillation_loss(source, target, margin):
    target = torch.max(target, margin)
    loss = torch.nn.functional.mse_loss(source, target, reduction="none")
    loss = loss * ((source > target) | (target > 0)).float()
    return loss.sum()

def dist_loss(source, target):
    loss = torch.nn.functional.mse_loss(source, target, reduction="none")
    loss = loss * ((source > target) | (target > 0)).float()
    return loss.sum()

def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)

def get_margin_from_BN(bn):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for (s, m) in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)

    return torch.FloatTensor(margin).to(std.device)

def compute_fsp(g , f_size):
        fsp_list = []
        for i in range(f_size-1):
            bot, top = g[i], g[i + 1]
            b_H, t_H = bot.shape[2], top.shape[2]
            if b_H > t_H:
                bot = F.adaptive_avg_pool2d(bot, (t_H, t_H))
            elif b_H < t_H:
                top = F.adaptive_avg_pool2d(top, (b_H, b_H))
            else:
                pass

            # print('layer: ', i , ' bot before reshape view: ', bot.shape)
            # print('layer: ', (i + 1) , ' top before reshape view: ', top.shape)

            bot = bot.view(bot.shape[0], bot.shape[1], -1)
            top = top.view(top.shape[0], top.shape[1], -1)

            bot = bot.unsqueeze(1)
            top = top.unsqueeze(2)

            
            # print('layer: ', i , ' bot after reshape view: ', bot.shape)
            # print('layer: ', (i + 1) , ' top after reshape view: ', top.shape)

            # bot = torch.nn.functional.normalize(bot , dim = -1)
            # top = torch.nn.functional.normalize(top , dim = -1)
            
            fsp = (bot * top).mean(-1)

            # print('fsp: ', fsp.shape)

            fsp_list.append(fsp)

        return fsp_list

def compute_fsp_loss(s, t):
        return (s - t).pow(2).mean()

class Distiller(nn.Module):
    def __init__(self, t_net, s_net, args):
        super(Distiller, self).__init__()

        t_channels = t_net.get_channel_num()
        s_channels = s_net.get_channel_num()

        self.Connectors = nn.ModuleList([build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])

        teacher_bns = t_net.get_bn_before_relu()
        margins = [get_margin_from_BN(bn) for bn in teacher_bns]
        for i, margin in enumerate(margins):
            self.register_buffer('margin%d' % (i+1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

        self.t_net = t_net
        self.s_net = s_net
        self.args = args
        self.loss_divider = [8, 4, 2, 1, 1, 4*4]
        self.criterion = sim_dis_compute
        self.temperature = 1
        self.scale = 0.5

    def forward(self, x):

        t_feats, t_out = self.t_net.extract_feature(x)
        s_feats, s_out = self.s_net.extract_feature(x)
        feat_num = len(t_feats)

        
        
        fsp_loss = 0 
        if self.args.fsp_lambda is not None: # pairwise loss

          num_layers = len(t_feats)   
          new_num_channels = 70 #t_out.shape[1]

          for layer_idx in range(num_layers):
            in_channels_t = t_feats[layer_idx].shape[1]
            in_channels_s = s_feats[layer_idx].shape[1]
            # print('layer: ', layer_idx , ' t_feats[layer_idx] before: ', t_feats[layer_idx].shape)
            # print('layer: ', layer_idx , ' s_feats[layer_idx] before: ', s_feats[layer_idx].shape)
            teacher_layer = nn.Conv2d(in_channels=in_channels_t, out_channels=new_num_channels, kernel_size=1).cuda()
            student_layer = nn.Conv2d(in_channels=in_channels_s, out_channels=new_num_channels, kernel_size=1).cuda() 

            t_feats[layer_idx] = teacher_layer(t_feats[layer_idx])
            s_feats[layer_idx] = student_layer(s_feats[layer_idx])
            # print('layer: ', layer_idx , ' t_feats[layer_idx] after : ', t_feats[layer_idx].shape)
            # print('layer: ', layer_idx , ' s_feats[layer_idx] after : ', s_feats[layer_idx].shape)

          fsp_t_list = compute_fsp(t_feats , len(t_feats))
          fsp_s_list = compute_fsp(s_feats , len(s_feats))

          loss_group = ([compute_fsp_loss(s, t) for s, t in zip(fsp_s_list, fsp_t_list)])
          loss = sum(loss_group)
          # print('loss_group: ', loss_group)
          # print('loss_group_mean: ', sum(loss_group))
          fsp_loss =  self.args.fsp_lambda * loss


        
        kd_loss = fsp_loss
        return s_out, kd_loss
