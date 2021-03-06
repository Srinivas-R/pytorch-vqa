import sys
import os.path
import math
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import config
import data
import model
import utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def update_learning_rate(optimizer, iteration):
    lr = config.initial_lr * 0.5**(float(iteration) / config.lr_halflife)
    for param_group in optimizer.param_groups:
        if param_group['lr'] != config.bert_lr:
            param_group['lr'] = lr
total_iterations = 0


def run(net, loader, optimizer, tracker, train=False, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    answ = []
    idxs = []
    accs = []
    tracker_class, tracker_params = tracker.MeanMonitor, {}
    if train:
        net.train()
    else:
        net.eval()
    tq = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))

    log_softmax = nn.LogSoftmax(dim=1).cuda()
    for v, q_inputs, q_masks, a, idx in tq:
        v = v.to(device, non_blocking=True)
        q_inputs = q_inputs.to(device, non_blocking=True)
        q_masks = q_masks.to(device, non_blocking=True)
        a = a.to(device, non_blocking=True)
        
        out = net(v, q_inputs, q_masks)
        nll = -log_softmax(out)
        loss = (nll * a / 10).sum(dim=1).mean()
        acc = utils.batch_accuracy(out.data, a.data).cpu()

        if train:
            global total_iterations
            update_learning_rate(optimizer, total_iterations)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_iterations += 1

        # store information about evaluation of this minibatch
        _, answer = out.data.cpu().max(dim=1)
        answ.append(answer.view(-1))
        accs.append(acc.view(-1))
        idxs.append(idx.view(-1).clone())

        loss_tracker.append(loss.item())
        for a in acc:
            acc_tracker.append(a.item())
        fmt = '{:.4f}'.format
        tq.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

    answ = list(torch.cat(answ, dim=0))
    accs = list(torch.cat(accs, dim=0))
    idxs = list(torch.cat(idxs, dim=0))
    return answ, accs, idxs


def main():
    if len(sys.argv) > 1:
        name = ' '.join(sys.argv[1:])
    else:
        from datetime import datetime
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    target_name = os.path.join('logs', '{}.pth'.format(name))
    print('will save to {}'.format(target_name))

    cudnn.benchmark = True

    train_loader = data.get_loader(train=True)
    val_loader = data.get_loader(val=True)

    net = nn.DataParallel(model.Net()).cuda()
    optim_params= [ {'params' : net.module.text.parameters(), 'lr' : config.initial_lr},
                    {'params' : net.module.attention.parameters(), 'lr' : config.initial_lr},
                    {'params' : net.module.classifier.parameters(), 'lr' : config.initial_lr}]
    optimizer = optim.Adam(optim_params)

    tracker = utils.Tracker()
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}

    for i in range(config.epochs):
        l = run(net, train_loader, optimizer, tracker, train=True, prefix='train', epoch=i)
        r = run(net, val_loader, optimizer, tracker, train=False, prefix='val', epoch=i)

        results = {
            'name': name,
            'tracker': tracker.to_dict(),
            'config': config_as_dict,
            'weights': net.state_dict(),
            'eval': {
                'answers': r[0],
                'accuracies': r[1],
                'idx': r[2],
            },
            'train': {
                'answers': l[0],
                'accuracies': l[1],
                'idx': l[2],
            },
            'vocab': train_loader.dataset.vocab,
        }
        torch.save(results, target_name)


if __name__ == '__main__':
    main()
