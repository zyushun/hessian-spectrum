import shutil
import torch
import os

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, savename='checkpoint.pth'):
    if not os.path.isdir('checkpoint'): # if no file named checkpoint, just create one
        os.mkdir('checkpoint')
    
    if is_best:
        torch.save(state, './checkpoint/'+str(savename)+'.pth')


def save_completecheckpoint(state, savename='checkpoint.pth'):
    if not os.path.isdir('checkpoint'):  # if no file named checkpoint, just create one
        os.mkdir('checkpoint')

    torch.save(state, './checkpoint/' + str(savename) + '.pth')


def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = init_lr * (0.1 ** (epoch // 30))

    if hasattr(optimizer, 'param_groups'): 
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # if hasattr(optimizer, 'lr'): 
    #     optimizer.lr = lr

    else: pass




def adjust_box(optimizer, epoch, init_box, init_boxtwo):


    
    box = init_box * (0.1 ** (epoch // 30))

    # print('iniital box', init_box)
    # print('epoch', epoch)
    # print('box',box)

    boxtwo = init_boxtwo * (0.1 ** (epoch // 30))

    if hasattr(optimizer, 'box_upperbound'): 
       optimizer.box_upperbound = box
    
    if hasattr(optimizer, 'box_upperbound_two'): 
       optimizer.box_upperbound_two = boxtwo
    else: pass

