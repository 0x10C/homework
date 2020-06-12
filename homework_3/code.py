import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology

class PASSRnet(nn.Module):
    def __init__(self, upscale_factor):
        super(PASSRnet, self).__init__()
        ### feature extraction
        self.init_feature = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            ResB(64),
            ResASPPB(64),
            ResB(64),
            ResASPPB(64),
            ResB(64),
        )
        ### paralax attention
        self.pam = PAM(64)
        ### upscaling
        self.upscale = nn.Sequential(
            ResB(64),
            ResB(64),
            ResB(64),
            ResB(64),
            nn.Conv2d(64, 64 * upscale_factor ** 2, 1, 1, 0, bias=False),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(64, 3, 3, 1, 1, bias=False),
            nn.Conv2d(3, 3, 3, 1, 1, bias=False)
        )
    def forward(self, x_left, x_right, is_training):
        ### feature extraction
        buffer_left = self.init_feature(x_left)
        buffer_right = self.init_feature(x_right)
        if is_training == 1:
            ### parallax attention
            buffer, (M_right_to_left, M_left_to_right), (M_left_right_left, M_right_left_right), \
            (V_left_to_right, V_right_to_left) = self.pam(buffer_left, buffer_right, is_training)
            ### upscaling
            out = self.upscale(buffer)
            return out, (M_right_to_left, M_left_to_right), (M_left_right_left, M_right_left_right), \
                   (V_left_to_right, V_right_to_left)
        if is_training == 0:
            ### parallax attention
            buffer = self.pam(buffer_left, buffer_right, is_training)
            ### upscaling
            out = self.upscale(buffer)
            return out

class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
        )
    def __call__(self,x):
        out = self.body(x)
        return out + x

class ResASPPB(nn.Module):
    def __init__(self, channels):
        super(ResASPPB, self).__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 4, 4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_1 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 8, 8, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 4, 4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_2 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 8, 8, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv1_3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1, 1, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 4, 4, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_3 = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 8, 8, bias=False), nn.LeakyReLU(0.1, inplace=True))
        self.b_1 = nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=False)
        self.b_2 = nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=False)
        self.b_3 = nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=False)
    def __call__(self, x):
        buffer_1 = []
        buffer_1.append(self.conv1_1(x))
        buffer_1.append(self.conv2_1(x))
        buffer_1.append(self.conv3_1(x))
        buffer_1 = self.b_1(torch.cat(buffer_1, 1))

        buffer_2 = []
        buffer_2.append(self.conv1_2(buffer_1))
        buffer_2.append(self.conv2_2(buffer_1))
        buffer_2.append(self.conv3_2(buffer_1))
        buffer_2 = self.b_2(torch.cat(buffer_2, 1))

        buffer_3 = []
        buffer_3.append(self.conv1_3(buffer_2))
        buffer_3.append(self.conv2_3(buffer_2))
        buffer_3.append(self.conv3_3(buffer_2))
        buffer_3 = self.b_3(torch.cat(buffer_3, 1))

        return x + buffer_1 + buffer_2 + buffer_3

class PAM(nn.Module):
    def __init__(self, channels):
        super(PAM, self).__init__()
        self.b1 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b2 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.b3 = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.softmax = nn.Softmax(-1)
        self.rb = ResB(64)
        self.fusion = nn.Conv2d(channels * 2 + 1, channels, 1, 1, 0, bias=True)
    def __call__(self, x_left, x_right, is_training):
        b, c, h, w = x_left.shape
        buffer_left = self.rb(x_left)
        buffer_right = self.rb(x_right)

        ### M_{right_to_left}
        Q = self.b1(buffer_left).permute(0, 2, 3, 1)                                                # B * H * W * C
        S = self.b2(buffer_right).permute(0, 2, 1, 3)                                               # B * H * C * W
        score = torch.bmm(Q.contiguous().view(-1, w, c),
                          S.contiguous().view(-1, c, w))                                            # (B*H) * W * W
        M_right_to_left = self.softmax(score)

        ### M_{left_to_right}
        Q = self.b1(buffer_right).permute(0, 2, 3, 1)                                               # B * H * W * C
        S = self.b2(buffer_left).permute(0, 2, 1, 3)                                                # B * H * C * W
        score = torch.bmm(Q.contiguous().view(-1, w, c),
                          S.contiguous().view(-1, c, w))                                            # (B*H) * W * W
        M_left_to_right = self.softmax(score)

        ### valid masks
        V_left_to_right = torch.sum(M_left_to_right.detach(), 1) > 0.1
        V_left_to_right = V_left_to_right.view(b, 1, h, w)                                          #  B * 1 * H * W
        V_left_to_right = morphologic_process(V_left_to_right)
        if is_training==1:
            V_right_to_left = torch.sum(M_right_to_left.detach(), 1) > 0.1
            V_right_to_left = V_right_to_left.view(b, 1, h, w)                                      #  B * 1 * H * W
            V_right_to_left = morphologic_process(V_right_to_left)

            M_left_right_left = torch.bmm(M_right_to_left, M_left_to_right)
            M_right_left_right = torch.bmm(M_left_to_right, M_right_to_left)

        ### fusion
        buffer = self.b3(x_right).permute(0,2,3,1).contiguous().view(-1, w, c)                      # (B*H) * W * C
        buffer = torch.bmm(M_right_to_left, buffer).contiguous().view(b, h, w, c).permute(0,3,1,2)  #  B * C * H * W
        out = self.fusion(torch.cat((buffer, x_left, V_left_to_right), 1))

        ## output
        if is_training == 1:
            return out, \
               (M_right_to_left.contiguous().view(b, h, w, w), M_left_to_right.contiguous().view(b, h, w, w)), \
               (M_left_right_left.view(b,h,w,w), M_right_left_right.view(b,h,w,w)), \
               (V_left_to_right, V_right_to_left)
        if is_training == 0:
            return out

def morphologic_process(mask):
    device = mask.device
    b,_,_,_ = mask.shape
    mask = 1-mask
    mask_np = mask.cpu().numpy().astype(bool)
    mask_np = morphology.remove_small_objects(mask_np, 20, 2)
    mask_np = morphology.remove_small_holes(mask_np, 10, 2)
    for idx in range(b):
        buffer = np.pad(mask_np[idx,0,:,:],((3,3),(3,3)),'constant')
        buffer = morphology.binary_closing(buffer, morphology.disk(3))
        mask_np[idx,0,:,:] = buffer[3:-3,3:-3]
    mask_np = 1-mask_np
    mask_np = mask_np.astype(float)

    return torch.from_numpy(mask_np).float().to(device)
  
  
  # -*- coding: utf-8 -*-
"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

from __future__ import print_function, division

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    iou = mean(ious)    # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []
        for i in range(C):
            if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [mean(iou) for iou in zip(*ious)] # mean accross images if per_image
    return 100 * np.array(ious)


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
         super(StableBCELoss, self).__init__()
    def forward(self, input, target):
         neg_abs = - input.abs()
         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
         return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


# --------------------------- MULTICLASS LOSSES ---------------------------


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float() # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n
  
  
  from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import *
import argparse
from tqdm import tqdm
import segmentation_models_pytorch as smp
import os
import logging
os.environ['CUDA_VISIBLE_DEVICES'] = '1,0'
from tensorboardX import SummaryWriter
import apex
from functools import partial
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default='efficientnet-b7')
    parser.add_argument("--problem", type=str, default='simple')
    parser.add_argument("--ENCODER", type=str, default='efficientnet-b7')
    parser.add_argument("--ENCODER_WEIGHTS", type=str, default='imagenet')
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-2, help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.2, help='')
    parser.add_argument('--n_epochs', type=int, default=1000, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=100, help='number of epochs to update learning rate')
    parser.add_argument('--dataset_dir', type=str, default='./data')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--pretrain_path', type=str, default='')
    parser.add_argument('--save_val_result', type=bool, default=True)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--focal_gamma', type=float, default=2.5)
    parser.add_argument('--dice_alpha', type=float, default=2)

    return parser.parse_args()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2.5, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        targets = 0.99 * targets + 0.01 * (1 - targets)
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        # alpha_t = torch.where(targets == 1, torch.tensor(self.alpha).cuda(), torch.tensor(1-self.alpha).cuda())
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

class DiceLoss_Fn(nn.Module):
    def __init__(self, sample_wise=False, use_focal=False, apply_nonlin=False, gamma=1.5):
        super(DiceLoss_Fn, self).__init__()
        self.sample_wise = sample_wise
        self.use_focal = use_focal
        self.apply_nonlin = apply_nonlin
        self.act = nn.Sigmoid()
        self.gamma = gamma

    def forward(self, input, label):
        label = 0.99 * label + 0.01 * (1 - label)
        eps = 1e-10
        if self.apply_nonlin:
            input = self.act(input)
        if not self.sample_wise:
            return 1 - (2 * (input * label).sum() + eps) / (input.sum() + label.sum() + eps)
        mul = torch.einsum('nchw->n', input*label)
        sum1 = torch.einsum('nchw->n', input)
        sum2 = torch.einsum('nchw->n', label)
        loss_sample_wise = 1 - (2 * mul / (sum1 + sum2)).mean()
        if not self.use_focal:
            return loss_sample_wise
        return (loss_sample_wise ** self.gamma).mean()

def test(net, test_loader, cfg, epoch):
    cudnn.benchmark = True
    net.eval()
    f1_epoch = []
    with torch.no_grad():
        for idx_iter, data in enumerate(test_loader):
            imgs, masks = data['image'], data['mask']
            imgs, masks = imgs.to(cfg.device, dtype=torch.float32), masks.to(cfg.device, dtype=torch.float32)
            imgs_name = test_loader.dataset.file_list[idx_iter].split('/')[-1]

            predict_masks = net(imgs)
            predict_masks = torch.sigmoid(predict_masks)
            f1_epoch.append(cal_f1(predict_masks, masks))
            if cfg.save_val_result:
                ## save results
                if not os.path.exists('log/'+cfg.name + '/epoch_{}'.format(epoch)):
                    os.mkdir('log/'+cfg.name+'/epoch_{}'.format(epoch))

                predict_masks = np.array(torch.squeeze(predict_masks.cpu(), 0))
                predict_masks = np.where(predict_masks > 0.5, 0, 255)
                predict_masks = predict_masks.transpose((1, 2, 0))
                cv2.imwrite('log/'+cfg.name+'/epoch_{}'.format(epoch) + '/'+imgs_name + '.tiff',  predict_masks)

        ## print results
        mean_f1 = float(np.array(f1_epoch).mean())
        logger.info('epoch_{} mean f1: {}'.format(epoch, mean_f1))
        tensorboard_writer.add_scalar('f1', mean_f1, epoch)
    net.train()
    return mean_f1


def train(train_loader, cfg):
    # net = UNet(n_channels=3, n_classes=1)
    # net = NestedUNet()
    # net.apply(weights_init_xavier)
    net = smp.Unet(
        encoder_name=cfg.ENCODER,
        encoder_weights=cfg.ENCODER_WEIGHTS,
        classes=1,
        activation=None,
        decoder_attention_type='scse',
        encoder_depth=5,
        decoder_channels=[1024, 512, 256, 128, 64],
        decoder_use_batchnorm=True
    )
    # net = convert_model(net)
    net = apex.parallel.convert_syncbn_model(net)
    logger.info(net)
    logger.info('parameters: {}'.format(sum(map(lambda x: x.numel(), net.parameters()))))
    net.to(cfg.device)
    if os.path.exists(cfg.pretrain_path):
        logger.info('load weight from {}'.format(cfg.pretrain_path))
        pretrained_dict = torch.load(cfg.pretrain_path)
        net.load_state_dict(pretrained_dict)
    cudnn.benchmark = True

    # criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5], dtype=torch.float32).cuda())
    criterion = FocalLoss(gamma=cfg.focal_gamma)
    # criterion_1 = DiceLoss_Fn()
    criterion_1 = smp.utils.losses.DiceLoss(activation='sigmoid')

    mom = 0.9
    alpha = 0.99
    eps = 1e-6
    if cfg.optimizer == 'Adam':
        logger.info('use Adam')
        optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    elif cfg.optimizer == 'RangerLars':
        logger.info('use RangerLars')
        from over9000.over9000 import Over9000
        optimizer = partial(Over9000, betas=(mom, alpha), eps=eps)
        optimizer = optimizer([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    elif cfg.optimizer == 'Novograd':
        logger.info('use Novograd')
        from over9000.novograd import Novograd
        optimizer = partial(Novograd, betas=(mom, alpha), eps=eps)
        optimizer = optimizer([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    elif cfg.optimizer == 'Ralamb':
        logger.info('use Ralamb')
        from over9000.ralamb import Ralamb
        optimizer = partial(Ralamb,  betas=(mom,alpha), eps=eps)
        optimizer = optimizer([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    elif cfg.optimizer == 'LookaheadAdam':
        logger.info('use LookaheadAdam')
        from over9000.lookahead import LookaheadAdam
        optimizer = partial(LookaheadAdam, betas=(mom, alpha), eps=eps)
        optimizer = optimizer([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    elif cfg.optimizer == 'Ranger':
        logger.info('use Ranger')
        from over9000.ranger import Ranger
        optimizer = partial(Ranger,  betas=(mom,alpha), eps=eps)
        optimizer = optimizer([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    else:
        raise NameError
    # if not cfg.use_Radam:
    #     logger.info('use adam')
    #     optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    # else:
    #     logger.info('use Radam')
    #     optimizer = RAdam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    # 半精度
    # net, optimizer = apex.amp.initialize(net, optimizer, opt_level="O1")
    net = nn.DataParallel(net)


    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=cfg.gamma, patience=30, verbose=True)
    loss_epoch = []
    f1_epoch = []
    best_f1 = 0
    iter_per_epoch = len(train_loader)
    accumulation_steps = 1
    for idx_epoch in range(cfg.n_epochs):
        net.train()
        for idx_iter, data in tqdm(enumerate(train_loader)):
            total_idx_iter = idx_epoch * iter_per_epoch + idx_iter + 1
            imgs, masks = data['image'], data['mask']
            imgs, masks = imgs.to(cfg.device, dtype=torch.float32), masks.to(cfg.device,dtype=torch.float32)
            predict_masks = net(imgs)

            loss = criterion(predict_masks, masks) + cfg.dice_alpha * criterion_1(predict_masks, masks)
            # loss = criterion_1(predict_masks, masks)

            # optimizer.zero_grad()
            # # with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
            # #     scaled_loss.backward()
            # loss.backward()
            # optimizer.step()
            loss.backward()

            if total_idx_iter % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

                loss_epoch.append(loss.data.cpu())
                f1_epoch.append(cal_f1(torch.sigmoid(predict_masks), masks))

        mean_loss = float(np.array(loss_epoch).mean())
        mean_f1 = float(np.array(f1_epoch).mean())
        logger.info('Epoch:{:5d} lr:{}, loss:  {:5f}, f1:  {:5f}'.format(idx_epoch + 1, optimizer.param_groups[0]['lr'],  mean_loss, mean_f1))

        scheduler.step(float(np.array(loss_epoch).mean()))
        loss_epoch = []
        f1_epoch = []
        if (idx_epoch+1) % 5 == 0:
            preprocessing_fn = smp.encoders.get_preprocessing_fn(cfg.ENCODER, "imagenet")
            test_set = ValSetLoader(dataset_dir=os.path.join(cfg.dataset_dir, cfg.problem, 'val'), cfg=cfg, preprocessing=get_preprocessing(preprocessing_fn))
            test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
            temp_f1 = test(net, test_loader, cfg, idx_epoch + 1)
            if temp_f1 > best_f1:
                save_ckpt(net,
                          path=os.path.join('./log', '{}'.format(cfg.name), 'ckpt'),
                          save_filename='best_ckpt.pth')
                line = 'best_f1: {} in {} epoch'.format(temp_f1, idx_epoch+1)
                filename = os.path.join('./log', '{}'.format(cfg.name), 'ckpt', 'msg.txt')
                with open(filename, 'w') as f:
                    f.write(line)
                best_f1 = temp_f1


# --------------------------- DATA PROCESSING FUNCTIONS ---------------------------
train_path = '/root/lyz/tsinghua/hw_2/train/'
test_path = '/root/lyz/tsinghua/hw_2/test/'
train = pd.read_csv(train_path+"feats.csv")
test = pd.read_csv(test_path+"feats.csv")
train.P53 = train.P53.astype(int)
test.P53 = test.P53.astype(int)
Y= train['molecular_subtype']
X = train[['age','HER2','P53']]
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=1, train_size=0.7)
clf2 = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, bootstrap=True)
clf2.fit(X,Y)
test_x = test[['age','HER2','P53']]
predict_y = clf2.predict(test_x)
result = pd.DataFrame()
result['id'] = test.id
result['result'] = predict_y
result.to_csv("rlt.csv",header = False,index=False)

                

if __name__ == '__main__':
    cfg = parse_args()
    if not os.path.exists(os.path.join('./log', '{}'.format(cfg.name))):
        os.makedirs(os.path.join('./log', '{}'.format(cfg.name)))
    setup_logger('base', os.path.join('./log', '{}'.format(cfg.name), '{}.log'.format(cfg.name)), level=logging.INFO,
                       screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(cfg)

    preprocessing_fn = smp.encoders.get_preprocessing_fn(cfg.ENCODER, "imagenet")
    train_set = TrainSetLoader(dataset_dir=os.path.join(cfg.dataset_dir, cfg.problem, 'train'), cfg=cfg, preprocessing=get_preprocessing(preprocessing_fn))
    logger.info('total {}, {} iter per epoch'.format(len(train_set), len(train_set) // cfg.batch_size))
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=cfg.batch_size, shuffle=True)

    tensorboard_log_dir = os.path.join('./tensorboard_log', cfg.name)
    if not os.path.exists(tensorboard_log_dir):
        os.makedirs(tensorboard_log_dir)
    tensorboard_writer = SummaryWriter(log_dir=tensorboard_log_dir)
    if cfg.ENCODER_WEIGHTS == 'None':
        logger.info('set cfg.ENCODER_WEIGHTS = None')
        cfg.ENCODER_WEIGHTS = None
    train(train_loader, cfg)
    
    
