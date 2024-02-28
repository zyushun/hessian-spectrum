import argparse
import os
import time
import shutil
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
import torch.backends.cudnn as cudnn
import random
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

from logger import Logger
from models import *
from data_loader import data_loader
from helper import AverageMeter, save_checkpoint,save_completecheckpoint, accuracy, adjust_learning_rate, adjust_box

from utils import progress_bar

import yaml
import io_utils
from torch.utils.tensorboard import SummaryWriter

import hessian_spectrum
import timm


model_names = [
    'alexnet', 'squeezenet1_0', 'squeezenet1_1', 'densenet121',
    'densenet169', 'densenet201', 'densenet201', 'densenet161',
    'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19', 'vgg19_bn', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'vit_base'
]

print("init amp")
scaler = torch.cuda.amp.GradScaler()

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='/home/yszhang/datasets/imagenet/', help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: alexnet)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='numer of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful to restarts)')
parser.add_argument('-b', '--batchsize', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR',
                    help='initial learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='beta1 of adam')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2 of adam')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum for sgd')
parser.add_argument('--wd', '--wd', default=1e-4, type=float,
                    metavar='W', help='Weight decay (default: 1e-4)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-m', '--pin-memory', dest='pin_memory', action='store_false',
                    help='use pin memory')
parser.add_argument('-p', '--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--print-freq', '-f', default=10, type=int, metavar='N',
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint, (default: None)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--opt', default='adam', type=str, help=' optimizer, adam or sgd or others')
parser.add_argument('--seed', default=32, type=int,help='seed for training. ')
parser.add_argument('--epsilon', '-eps', default=1e-8, type=float, help='epsilon for stability')
parser.add_argument('--comment', '-comment', default='-', type=str, help='some additional comments')
parser.add_argument('--use_minibatch', action='store_true', help='Set the flag to True')

parser.add_argument('--load_iter', type = int, default=0, help='load ')

parser.add_argument('--gradient_accumulation_steps', type = int, default = 0 )
parser.add_argument('--shuffle',action='store_true', help = 'whether use shuffle in training data.')
parser.add_argument('--plot_hessian',action='store_true', help = 'whether plot hessian or not')
# best_prec1 = 0.0

def main():
    global args #, best_prec1
    args = parser.parse_args()

    #print('worldsize',int(os.environ["WORLD_SIZE"]))

    print('w',os.environ.get('WORLD_SIZE'))
    print('w2', int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1)
    print('ngpus_per_node = torch.cuda.device_count()',torch.cuda.device_count())


    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        print("=> creating model '{}'".format(args.arch))
    last_time = time.time()
    if args.arch == 'alexnet':
        model = alexnet(pretrained=args.pretrained)
    elif args.arch == 'squeezenet1_0':
        model = squeezenet1_0(pretrained=args.pretrained)
    elif args.arch == 'squeezenet1_1':
        model = squeezenet1_1(pretrained=args.pretrained)
    elif args.arch == 'densenet121':
        model = densenet121(pretrained=args.pretrained)
    elif args.arch == 'densenet169':
        model = densenet169(pretrained=args.pretrained)
    elif args.arch == 'densenet201':
        model = densenet201(pretrained=args.pretrained)
    elif args.arch == 'densenet161':
        model = densenet161(pretrained=args.pretrained)
    elif args.arch == 'vgg11':
        model = vgg11(pretrained=args.pretrained)
    elif args.arch == 'vgg13':
        model = vgg13(pretrained=args.pretrained)
    elif args.arch == 'vgg16':
        model = vgg16(pretrained=args.pretrained)
    elif args.arch == 'vgg19':
        model = vgg19(pretrained=args.pretrained)
    elif args.arch == 'vgg11_bn':
        model = vgg11_bn(pretrained=args.pretrained)
    elif args.arch == 'vgg13_bn':
        model = vgg13_bn(pretrained=args.pretrained)
    elif args.arch == 'vgg16_bn':
        model = vgg16_bn(pretrained=args.pretrained)
    elif args.arch == 'vgg19_bn':
        model = vgg19_bn(pretrained=args.pretrained)
    elif args.arch == 'resnet18':
        model = resnet18(pretrained=args.pretrained)
    elif args.arch == 'resnet34':
        model = resnet34(pretrained=args.pretrained)
    elif args.arch == 'resnet50':
        model = resnet50(pretrained=args.pretrained)
    elif args.arch == 'resnet101':
        model = resnet101(pretrained=args.pretrained)
    elif args.arch == 'resnet152':
        model = resnet152(pretrained=args.pretrained)
    elif args.arch == 'vit_base': # not working
        model = timm.create_model('vit_base_patch16_224',pretrained=False)
    else:
        raise NotImplementedError


    model.cuda()

    'parallel'
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True
    #model = torch.nn.parallel.DistributedDataParallel(model)


    #model = torch.nn.parallel.DistributedDataParallel(model)

    # define loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda()


    if args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum,
                          weight_decay=args.wd)
    elif args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon,
                               weight_decay=args.wd, amsgrad=False)

    elif args.opt == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon,
                                weight_decay=args.wd, amsgrad=False)
                                     
    # optionlly resume from a checkpoint


    if args.load_iter == 0:
        args.resume = ''

    if args.resume:
        file_name = args.resume + args.arch + args.opt+'_ckpt_'+str(args.load_iter)+'.pth'

        if os.path.isfile(file_name):
            print("=> loading checkpoint '{}'".format(file_name))
            checkpoint = torch.load(file_name)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(file_name, checkpoint['epoch']))
        
            print('resumed from: ', file_name)
            time.sleep(3)

        else:
            print("=> no checkpoint found")
            assert 0





    os.makedirs('log', exist_ok= True)
    'set up logger: change name here'
    save_dir = os.path.join(
        'log',
        'ImageNET-model-{}-opt-{}-lr-{}-beta1-{}-beta2-{}-bs-{}-seed-{}-comment-{}'.format(
             args.arch, args.opt, args.lr, args.beta1, args.beta2, args.batchsize, args.seed, args.comment)
    )



    os.makedirs(save_dir, exist_ok=True)
    logger = Logger('{}/logger.txt'.format(save_dir), title='logger')
    logger.set_names(['epoch', 'trainloss', 'testloss','trainacc','testacc'])
    writer = SummaryWriter(save_dir)
    io_utils.save_code(save_dir)


    # cudnn.benchmark = True

    # Data loading
    print('start loading data')
    ngpus_per_node = torch.cuda.device_count()
    #args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    #print('load data worker',args.workers)
    print('load data pin_mem', args.pin_memory)


    if args.plot_hessian:
        args.shuffle = False
    else: args.shuffle = True

    train_loader, val_loader = data_loader(args.data, args.batchsize, args.workers, args.pin_memory, shuffle = args.shuffle)
    print('loading complete')


    if args.evaluate:
        test(val_loader, model)
        return
    
    if args.plot_hessian:
        plot_hessian(args, model, train_loader )

    else: training(args, optimizer, save_dir, train_loader, val_loader, model, criterion, writer,logger)


   
def plot_hessian(args, model, train_loader ):

    batch_size = args.batchsize

    gradient_accumulation_steps = args.gradient_accumulation_steps # 256 *30

    load_iter = args.load_iter


    comment = args.comment

    #vit 
    sample_layer = [
        "module.patch_embed.proj.weight",
        "module.blocks.0.attn.qkv.weight",
        "module.blocks.0.attn.proj.weight",
        "module.blocks.0.mlp.fc1.weight",
        "module.blocks.0.mlp.fc2.weight",
        "module.blocks.6.attn.qkv.weight",
        "module.blocks.6.attn.proj.weight",
        "module.blocks.6.mlp.fc1.weight",
        "module.blocks.6.mlp.fc2.weight",
        "module.blocks.11.attn.qkv.weight",
        "module.blocks.11.attn.proj.weight",
        "module.blocks.11.mlp.fc1.weight",
        "module.blocks.11.mlp.fc2.weight",
        "module.head.weight"
    ]


    hessian = hessian_spectrum.Hessian(model = model, ckpt_iteration = load_iter, use_minibatch= args.use_minibatch, gradient_accumulation_steps = gradient_accumulation_steps, train_data = train_loader, batch_size= batch_size,sample_layer = sample_layer, comment = comment)


    hessian.get_spectrum(layer_by_layer = True)
    hessian.load_curve(layer_by_layer = True)


    hessian.get_spectrum(layer_by_layer = False)
    hessian.load_curve(layer_by_layer = False)





def training(args, optimizer, save_dir, train_loader, val_loader, model, criterion, writer,logger):
    for epoch in range(args.start_epoch, args.epochs):
    
        adjust_learning_rate(optimizer, epoch, args.lr)
        print('Epoch-{}'.format(epoch))
        print('task', save_dir)

        trainloss, trainacc =train(train_loader, model, criterion, optimizer, epoch, args.print_freq)


        testloss, testacc = test(val_loader, model)

        print("trainloss={}trainacc={}testloss={}testacc={}".format(trainloss,trainacc,testloss,testacc))

        writer.add_scalar('trainloss', trainloss, epoch)
        writer.add_scalar('trainacc', trainacc, epoch)
        writer.add_scalar('testloss', testloss, epoch)
        writer.add_scalar('testacc', testacc, epoch)
        logger.append([epoch, trainloss, testloss, trainacc, testacc])

        # remember the best prec@1 and save checkpoint

        if epoch % 5 == 1 or epoch == 1 or epoch == args.epochs -1: 
            print('saving checkpoint..')
            save_completecheckpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, savename = args.arch + args.opt + '_ckpt_' + str(epoch))
    

    logger.close()
    logger.plot()
    io_utils.save_code(save_dir)
    yaml.safe_dump(args.__dict__, open(os.path.join(save_dir, 'config.yml'), 'w'), default_flow_style=False)




def train(train_loader, model, criterion, optimizer, epoch, print_freq):
    
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    train_loss = 0
    correct = 0
    total = 0

    # switch to train mode
    model.train()

    end = time.time()

 
    for batch_idx, (inputs, targets) in enumerate(train_loader):

        targets = targets.cuda()
        inputs = inputs.cuda()
        'regular update'
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()        
        optimizer.step()
        optimizer.zero_grad()


        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()



        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
 

    trainacc=correct/total
    trainloss=train_loss/(batch_idx+1)




    return trainloss, trainacc



def test(val_loader, model):
    
    criterion = nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            #
            progress_bar(batch_idx, len(val_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))


    testloss=test_loss/(batch_idx+1)

    testacc = correct/total


    return testloss, testacc





if __name__ == '__main__':
    main()
