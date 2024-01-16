import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch.cuda.amp
from conf import settings
from models.MS_ResNet_TT import tt_resnet18, tt_resnet34, tt_resnet20
from models.MS_ResNet import resnet18, resnet34, resnet20
from data_loader import cifar10_loader, cifar100_loader, build_ncaltech
from spikingjelly.datasets.n_caltech101 import NCaltech101
from torch.utils.data import Dataset, DataLoader


def train(epoch, args):
    running_loss = 0
    start = time.time()
    net.cuda()
    net.train()
    correct = 0.0
    num_sample = 0
    clock = [0]
    wallclock = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for batch_index, (images, labels) in enumerate(trainloader):
        if args.gpu:
            labels = labels.cuda(non_blocking=True)
            images = images.cuda(non_blocking=True)

        if 'CIFAR' not in args.dataset:
            images = images.transpose(1,0)
        num_sample += images.size()[0]
        optimizer.zero_grad()

        # wallcolck measurement
        wallclock.record()
        outputs = net(images)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        loss = loss_function(outputs, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()  
        end.record()
        torch.cuda.synchronize()
        clock.append(wallclock.elapsed_time(end))

        n_iter = (epoch - 1) * len(trainloader) + batch_index + 1
        if batch_index % 10 == 9:
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                running_loss/10,
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * args.b + len(images),
                total_samples=len(trainloader.dataset)
            ))
            print('1 batch wallclock time consumed: {:.2f}s'.format(np.mean(clock)))
            print('training time consumed: {:.2f}s'.format(time.time() - start)) 
            # if args.local_rank == 0:
            writer.add_scalar('Train/avg_loss', running_loss/10, n_iter)
            writer.add_scalar('Train/avg_loss_numpic', running_loss/10, n_iter * args.b)
            running_loss = 0
    finish = time.time()
    # if args.local_rank == 0:
    writer.add_scalar('Train/acc', correct/num_sample*100, epoch)
    print("Training accuracy: {:.2f} of epoch {}".format(correct/num_sample*100, epoch))
    print('epoch {} wallclock time consumed: {:.2f}s'.format(epoch, np.mean(clock)))
    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))


@torch.no_grad()
def eval_training(epoch, args):
    start = time.time()
    net.eval()

    test_loss = 0.0
    correct = 0.0
    real_batch = 0
    for (images, labels) in testloader:
        
        real_batch += images.size()[0]
        if args.gpu:            
            
            images = images.cuda()
            labels = labels.cuda()
        if 'CIFAR' not in args.dataset:
            images = images.transpose(1,0)
        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    print('Evaluating Network.....')
    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}%, Time consumed:{:.2f}s'.format(
        test_loss * args.b / len(testloader.dataset),
        correct.float() / real_batch * 100,
        finish - start
        ))

    # if args.local_rank == 0:
        # add information to tensorboard
    writer.add_scalar('Test/Average loss', test_loss * args.b / len(testloader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / real_batch * 100, epoch)

    return correct.float() / len(testloader.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=str, required=True, help='net type')
    parser.add_argument('--dataset', type=str, required=True, help='[CIFAR10, CIFAR100, N_caltech101]')
    parser.add_argument('--gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('--b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--T', type=int, default=4, help='timestep')
    parser.add_argument('--tt_mode', type=str, default='PTT', help='[STT / PTT / HTT]')
    parser.add_argument('--rank_path', type=str, help='checkpoint of TT-ranks')
    args = parser.parse_args()
    torch.cuda.set_device(0)

    SEED = 445
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)


    if args.dataset == 'CIFAR10':
        train_dataset, test_dataset = cifar10_loader(args)
        num_class = 10
        LOG_INFO = "CIFAR10"
    elif args.dataset == 'CIFAR100':
        train_dataset, test_dataset = cifar100_loader(args)
        num_class = 100
        LOG_INFO = "CIFAR100"
    elif args.dataset == 'N_caltech101':
        dataset = NCaltech101(root='dataset/n-caltech', data_type='frame', frames_number=6, split_by='number')
        train_dataset, test_dataset = build_ncaltech(args)
        num_class = 101
        LOG_INFO = "N_caltech101"
    else:
        raise NotImplementedError

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.b, shuffle=True, num_workers=8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.b, shuffle=False, num_workers=8, pin_memory=True)
    timestep = args.T
    
    if args.tt_mode:
        # to load a pretrained model
        if args.depth == '18':
            Path = args.rank_path
        elif args.depth == '34':
            Path = args.rank_path
        checkpoint = torch.load(Path)
        rank_list = checkpoint['rankList']
        print(len(rank_list),rank_list)

        print('TT imported!')
        args.net = "TT-resnet" + args.depth
        if args.depth == "18":
            net = tt_resnet18(rank_list, num_class)
        elif args.depth == "34":
            net = tt_resnet34(rank_list, num_class)

    else:
        args.net = "ori-resnet" + args.depth
        print('orginal imported')
        if args.depth == "18":
            net = resnet18(num_class)
        elif args.depth == "34":
            net = resnet34(num_class)

    net.cuda()   
    

    # learning rate should go with batch size.
    b_lr = args.lr
    loss_function = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.SGD([{'params': net.parameters(), 'initial_lr': b_lr}], momentum=0.9, lr=b_lr, weight_decay=1e-4)
    train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.EPOCH, eta_min=0, last_epoch=0)
    iter_per_epoch = len(trainloader)
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, LOG_INFO, args.net, str(args.b), str(args.lr), settings.TIME_NOW)

    # use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    writer = SummaryWriter(log_dir=os.path.join(
        settings.LOG_DIR, args.net, str(args.b), str(args.lr), LOG_INFO, settings.TIME_NOW))

    # create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
    best_acc = 0.0
    for epoch in range(1, settings.EPOCH+1):
        settings.EPOCH_INDEX = epoch
        print(settings.EPOCH_INDEX)
        train(epoch, args)

        train_scheduler.step()
        acc = eval_training(epoch, args)
        if epoch > (settings.EPOCH-10) and best_acc < acc:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path.format(net=args.net, epoch=epoch, type='best'))
            best_acc = acc
            continue
        elif epoch >= (settings.EPOCH-10):
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))
            continue
        elif (not epoch % settings.SAVE_EPOCH):
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))
            continue

    writer.close()
