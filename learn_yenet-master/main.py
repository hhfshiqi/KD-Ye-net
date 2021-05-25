from __future__ import print_function
import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import student
import YeNet
import random
import numpy as np
import cv2
from glob import glob

parser = argparse.ArgumentParser(description='PyTorch implementation of YeNet')
parser.add_argument('--train_cover_dir', type=str, default="/media/s304/A/hhf/data/ww-zhongxin-0.5/train/cover/",
                    metavar='PATH',
                    help='path of directory containing all training cover images')
parser.add_argument('--train_stego_dir', type=str, default="/media/s304/A/hhf/data/ww-zhongxin-0.5/train/stego/",
                    metavar='PATH',
                    help='path of directory containing all training stego images or beta maps')
parser.add_argument('--valid_cover_dir', type=str, default="/media/s304/A/hhf/data/ww-zhongxin-0.5/valid/cover/",
                    metavar='PATH',
                    help='path of directory containing all validation cover images')
parser.add_argument('--valid_stego_dir', type=str, default="/media/s304/A/hhf/data/w-zhongxin-0.5/valid/stego/",
                    metavar='PATH',
                    help='path of directory containing all validation stego images or beta maps')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 32)')
parser.add_argument('--epochs', type=int, default=11, metavar='N', help='number of epochs to train (default: 1000)')
parser.add_argument('--lr', type=float, default=0.4, metavar='LR', help='learning rate (default: 4e-1)')
parser.add_argument('--use-batch-norm', action='store_true', default=False,
                    help='use batch normalization after each activation, also disable pair constraint (default: False)')
parser.add_argument('--embed-otf', action='store_true', default=False,
                    help='use beta maps and embed on the fly instead of use stego images (default: False)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--gpu', type=int, default=2, help='index of gpu used (default: 0)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--path', type=str, default="/media/s304/A/hhf/learn_yenet-master(复件)/0.3_checkpoints/model_best.pth.tar", metavar='path')
parser.add_argument("--results-dir", default='results', dest="results_dir", type=str,
                    help="Where all results are collected")
parser.add_argument("--teacher", default="t_net", type=str, dest="t_name", help="teacher student name")
parser.add_argument("--student", default="s_net", dest="s_name", type=str, help="teacher student name")

args = parser.parse_args()
arch = 'YeNet_with_bn' if args.use_batch_norm else 'YeNet'

args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(args.gpu)
else:
    args.gpu = None
kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

print("Generate loaders...")

print("Generate model")
# t_net = student.YeeNet(with_bn=args.use_batch_norm)
t_net = YeNet.Net()
s_net = student.YeeNet(with_bn=args.use_batch_norm)
print(s_net)
print("Generate loss and optimizer")
if args.cuda:
    t_net.cuda()
    s_net.cuda()

optimizer = optim.Adadelta(s_net.parameters(), lr=args.lr, rho=0.95, eps=1e-8, weight_decay=5e-4)
_time = time.time()


class ToTensor():
    def __call__(self, sample):
        data, label = sample['data'], sample['label']

        data = np.expand_dims(data, axis=1)
        data = data.astype(np.float32)
        # data = data / 255.0
        new_sample = {
            'data': torch.from_numpy(data),
            'label': torch.from_numpy(label).long(),
        }

        return new_sample


class RandomRot(object):
    def __call__(self, sample):
        data = sample['data']
        rot = random.randint(0, 3)
        return {'data': np.rot90(data, rot, axes=[1, 2]).copy(),
                'label': sample['label']}


class RandomFlip(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            data = sample['data']
            return {'data': np.flip(data, axis=2).copy(),
                    'label': sample['label']}
        else:
            return sample


train_transform = transforms.Compose([
    RandomRot(),
    RandomFlip(),
    ToTensor()
])

eval_transform = transforms.Compose([
    ToTensor()
])

class MyDataset(Dataset):
    def __init__(self, DATASET_DIR, transform=None):
        random.seed(1234)
        self.transform = transform
        self.cover_dir = DATASET_DIR + '/cover/'
        self.stego_dir = DATASET_DIR + '/stego/'
        self.cover_list = [x.split('/')[-1] for x in glob(self.cover_dir + '/*')]
        random.shuffle(self.cover_list)
        assert len(self.cover_list) != 0, "cover_dir is empty"

    def __len__(self):
        return len(self.cover_list)

    def __getitem__(self, idx):
        file_index = int(idx)
        cover_path = os.path.join(self.cover_dir, self.cover_list[file_index])
        stego_path = os.path.join(self.stego_dir, self.cover_list[file_index])
        cover_data = cv2.imread(cover_path, -1)
        stego_data = cv2.imread(stego_path, -1)
        data = np.stack([cover_data, stego_data])
        label = np.array([0, 1], dtype='int32')
        sample = {'data': data, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

train_path = "/media/s304/A/hhf/data/s-zi-0.3/train/"
valid_path = "/media/s304/A/hhf/data/s-zhongxin-0.3/valid/"
test_path = "/media/s304/A/hhf/data/s-zhongxin-0.3/train/"
train_dataset = MyDataset(train_path, train_transform)
valid_dataset = MyDataset(valid_path, eval_transform)
test_dataset = MyDataset(test_path, train_transform)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

print('train_loader have {} iterations, valid_loader have {} iterations'.format(
    len(train_loader), len(valid_loader)))

loss_fun = nn.CrossEntropyLoss().cuda()


def criterion(out_s, out_t, target, T, lambda_):
    #    loss_fun = nn.CrossEntropyLoss().cuda()
    kd_fun = nn.KLDivLoss(size_average=False).cuda()
    loss = loss_fun(out_s, target)
    batch_size = target.shape[0]
    s_max = F.log_softmax(out_s / T, dim=1)
    t_max = F.softmax(out_t / T, dim=1)
    loss_kd = kd_fun(s_max, t_max) / batch_size
    loss = (1 - lambda_) * loss + lambda_ * T * T * loss_kd
    return loss


def accuracy(outputs, labels):
    _, argmax = torch.max(outputs, 1)
    return (labels == argmax.squeeze()).float().mean()

def train(epoch):
    running_loss = 0.
    running_accuracy = 0.
    for batch_idx, sample in enumerate(train_loader):
        images, labels = sample['data'], sample['label']
        shape = list(images.size())
        images = images.reshape(shape[0] * shape[1], *shape[2:])
        labels = labels.reshape(-1)
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        t_out = t_net(images)
        s_out = s_net(images)
        # print(t_out)
        # print("...\n")
        # print(s_out)
        acc = accuracy(s_out, labels).item()
        running_accuracy += acc
        loss = criterion(s_out, t_out, labels, 4, 0.1)
        # loss = loss_fun(s_out, labels)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % args.log_interval == 0:
            running_accuracy /= args.log_interval
            running_loss /= args.log_interval
            print(('Train epoch: {} [{}/{}]\tAccuracy: ' +
                   '{:.2f}%\tLoss: {:.6f}').format(
                epoch, batch_idx + 1, len(train_loader),
                       100 * running_accuracy, running_loss))
            running_loss = 0.
            running_accuracy = 0.


def valid():
    valid_loss = 0.
    valid_accuracy = 0.
    correct = 0
    for sample in valid_loader:
        images, labels = sample['data'], sample['label']
        shape = list(images.size())
        images = images.reshape(shape[0] * shape[1], *shape[2:])
        labels = labels.reshape(-1)
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()
        outputs = s_net(images)
        valid_loss += loss_fun(outputs, labels).item()
        valid_accuracy += accuracy(outputs, labels).item()
        # valid_accuracy += YeNet.accuracy(outputs, labels)
    valid_loss /= len(valid_loader)
    valid_accuracy /= len(valid_loader)
    print('Test set: Loss: {:.4f}, Accuracy: {:.2f}%)\n'.format(
        valid_loss, 100 * valid_accuracy))
    return valid_loss, valid_accuracy


def test():
    test_loss = 0.
    test_accuracy = 0.
    for sample in test_loader:
        images, labels = sample['data'], sample['label']
        shape = list(images.size())
        images = images.reshape(shape[0] * shape[1], *shape[2:])
        labels = labels.reshape(-1)
        if args.cuda:
            images, labels = images.cuda(), labels.cuda()
        outputs = s_net(images)
        test_loss += loss_fun(outputs, labels).item()
        test_accuracy += accuracy(outputs, labels).item()
    test_loss /= len(test_loader)
    test_accuracy /= len(test_loader)
    print('\nTest set: Loss: {:.4f}, Accuracy: {:.2f}%)\n'.format(
        test_loss, 100 * test_accuracy))
    return test_loss, test_accuracy



def save_checkpoint(state, is_best, filename='stu_checkpoints/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'stu_checkpoints/model_best.pth.tar')




checkpoint = torch.load('/media/s304/A/hhf/结果模型保存/s-uniward/0.4/0.4_checkpoints/model_best.pth.tar')
t_net.load_state_dict(checkpoint['state_dict'])
# checkpoint1 = torch.load('stu_checkpoints/model_best.pth.tar')
# # checkpoint1 = torch.load('/media/s304/A/hhf/结果模型保存/s-uniward/0.2/stu_checkpoints/model_best.pth.tar')
# s_net.load_state_dict(checkpoint1['state_dict'])
# best_accuracy = checkpoint1['best_prec1']
# start_epoch = checkpoint1['epoch']+1
start_epoch = 1
best_accuracy = 0.
print("Time:", time.time() - _time)
test()
test()
test()
print("Time:", time.time() - _time)
for epoch in range(start_epoch, args.epochs + 1):
    print("Epoch:", epoch)
    print("Train")
    train(epoch)
    print("Time:", time.time() - _time)
    print("Test")
    _, accu = valid()
    if accu > best_accuracy:
        best_accuracy = accu
        is_best = True
    else:
        is_best = False
    print("Time:", time.time() - _time)
    save_checkpoint({
        'epoch': epoch,
        'arch': arch,
        'state_dict': s_net.state_dict(),
        'best_prec1': accu,
        'optimizer': optimizer.state_dict(),
    }, is_best)
test()
