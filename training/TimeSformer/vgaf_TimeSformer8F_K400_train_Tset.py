######## TimeSformer ###########
# !pip install timesformer-pytorch
import glob
import io
import logging
import math
import os
import pathlib
import pickle
import pprint
import random
import shutil
import sys
import time
from einops import rearrange
from collections import OrderedDict
from functools import partial
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms as transform
import tqdm
from PIL import Image, ImageOps, ImageEnhance
from sklearn.metrics import accuracy_score
from timesformer.models.vit import TimeSformer
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch.utils import data
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
# from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 2
num_frames = 8
N_EPOCHS = 20
model_name = "TimeSformer_8FK100"
training_set = "Tset" # Tset or TVset
onlySaveBestAccModel = False

run_identity = "vgaf_"+ model_name +str(num_frames)+"F_train_"+training_set
print("******************************************")
print("Experimental run for : " + run_identity)
if onlySaveBestAccModel:
    print("Note: Only best accuracy checkpoint in each epoch will be saved")
else:
    print("Note: checkpoint in each epoch will be saved")
print("******************************************")

if training_set == "TVset":
    train_videos_path = '../TrainValAll'
    val_videos_path = '../TestAll'
    train_videos_frames_path = '/scratch/aarti9/TrainVal_'+str(num_frames)+'_Frames'
    val_videos_frames_path = '/scratch/aarti9/Test_'+str(num_frames)+'_Frames'
else:
    train_videos_path = '../TrainAll'
    val_videos_path = '../ValAll'
    train_videos_frames_path = '/scratch/aarti9/Train_' + str(num_frames) + '_Frames'
    val_videos_frames_path = '/scratch/aarti9/Val_' + str(num_frames) + '_Frames'

SEED = 0
np.random.seed(SEED) # if numpy is used
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

output_dir = "logs"
os.makedirs(output_dir, exist_ok=True)
time_str = time.strftime('%Y-%m-%d-%H-%M')
log_file = '{}_{}.log'.format(run_identity, time_str)
final_log_file = os.path.join(output_dir, log_file)
head = '%(asctime)-15s %(message)s'
logging.basicConfig(filename=str(final_log_file),
                    format=head)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
console = logging.StreamHandler()
loss_func = nn.CrossEntropyLoss()





model = TimeSformer(img_size=224, num_classes=400, num_frames=8, attention_type='divided_space_time',  pretrained_model='TimeSformer_divST_8x32_224_K400.pyth')

parameters = filter(lambda p: p.requires_grad, model.parameters())
parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
print('Trainable Parameters: %.3fM' % parameters)


model.model.reset_classifier(10)  # This is for VGAF

model = model.cuda()
# Batchnorm parameters.
bn_params = []
# Non-batchnorm parameters.
non_bn_parameters = []
for name, p in model.named_parameters():
    if "bn" in name:
        bn_params.append(p)
    else:
        non_bn_parameters.append(p)

optim_params = [
        {"params": bn_params, "weight_decay": 0.0},
        {"params": non_bn_parameters, "weight_decay": 0.0001},
    ]

optimizer = torch.optim.SGD(
            optim_params,
            lr=0.0002,
            momentum=0.9,
            weight_decay=1e-4,
            dampening=0.0,
            nesterov=True,
        )

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def train_epoch(model, optimizer, data_loader, loss_history, loss_func, step, epoch):
    top1 = AverageMeter('acc@1', ':6.2f')
    top5 = AverageMeter('acc@5', ':6.2f')
    total_samples = len(data_loader.dataset)
    model.train()
    loader = tqdm.tqdm(data_loader, desc='Loading train data')
    avg_loss = 0
    for i, (data, target) in enumerate(loader):
        # scheduler(optimizer, i, epoch)
        optimizer.zero_grad()

        # data = rearrange(data, 'b c t h w -> (b t) c h w').cuda(0)
        data = rearrange(data, 'b p h w c -> b c p h w').cuda(0)
        # data = data.cuda(0)
        target = target.type(torch.LongTensor).cuda(0)

        pred = model(data.float())
        # print("shape of pred", pred.shape)
        loss = loss_func(pred, target)
        loss.backward()
        optimizer.step()
        prec1 = accuracy(pred, target, topk=(1,))
        top1.update(prec1[0].item(), pred.size(0))
        # top5.update(prec5.item(), pred.size(0))
        step += 1
        avg_loss += loss.item()
        if i % 5 == 0:
            loader.set_description(
                # 'train_loss: %.3f, acc1: %.3f, acc5: %.3f, step %d' %
                'train_loss: %.3f, acc1: %.3f, step %d' %
                # (avg_loss / (i + 1), top1.avg, top5.avg, step))
                (avg_loss / (i + 1), top1.avg, step))

            loss_history.append(loss.item())
    return step

def evaluate(model, data_loader, loss_history, loss_func, valid_acc_max, onlySaveBestAccModel):
    model.eval()
    top1 = AverageMeter('acc@1', ':6.2f')
    # top5 = AverageMeter('acc@5', ':6.2f')
    total_samples = len(data_loader.dataset)
    # correct_samples = 0
    total_loss = 0
    loader = tqdm.tqdm(data_loader, desc='\r')

    for i, (data, target) in enumerate(loader):
        # data = rearrange(data, 'b c t h w -> (b t) c h w').cuda(0)
        # data = data.cuda(0)
        data = rearrange(data, 'b p h w c -> b c p h w').cuda(0)

        target = target.type(torch.LongTensor).cuda(0)
        with torch.no_grad():
            pred = model(data.float())
            loss = loss_func(pred, target)
            # _, pred = torch.max(output, dim=1)
            prec1 = accuracy(pred, target, topk=(1,))
            top1.update(prec1[0].item(), pred.size(0))
            # top5.update(prec5.item(), pred.size(0))

            total_loss += loss.item()
            loader.set_description(
                'val_loss:   %.3f, acc1: %.3f' %
                (total_loss / (i + 1), top1.avg))

            val_acc = top1.avg
            if (not onlySaveBestAccModel) or (onlySaveBestAccModel and valid_acc_max <= val_acc):
                if valid_acc_max <= val_acc:
                    print('Validation acc increased ({:.5f} --> {:.5f}).  Saving model ...'.format(valid_acc_max, val_acc))
                print('Epoch - ' + str(epoch))

                torch.save(
                    {
                        "model": model.state_dict(),
                        "optim": optimizer.state_dict(),
                        "epoch": epoch
                    },
                    "/scratch/aarti9/model_chkpt_"+run_identity+"_E"+str(epoch)+".pt",
                )
                valid_acc_max = val_acc

    loss_history.append(total_loss / len(data_loader))
    print('val_loss:   %.3f, acc1: %.3f' % (total_loss / len(data_loader), top1.avg))
    logger.info('val_loss:   %.3f, acc1: %.3f' % (total_loss / len(data_loader), top1.avg))

    return valid_acc_max

class DatasetProcessing(data.Dataset):
    def __init__(self, videos_path, framespath):
        super(DatasetProcessing, self).__init__()
        #         List of all videos path
        video_list = []
        for root, dirs, files in os.walk(videos_path):
            for file in files:
                fullpath = os.path.join(root, file)
                if ('.mp4' in fullpath):
                    video_list.append(fullpath)
        self.video_list = np.asarray(video_list)
        self.framespath = framespath

    def __getitem__(self, index):
        video_name = self.video_list[index].split('/')[-1]
        video_name_noext = os.path.splitext(video_name)[0]
        with open(self.framespath + '/' + video_name_noext + '.pkl', 'rb') as f:
            w_list = pickle.load(f)

        return w_list[0], w_list[1]

    def __len__(self):
        return self.video_list.shape[0]

dset_train = DatasetProcessing(train_videos_path, train_videos_frames_path)
dset_val = DatasetProcessing(val_videos_path, val_videos_frames_path)

trainloader = DataLoader(dset_train,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)
valloader = DataLoader(dset_val,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=0)

train_loss_history, test_loss_history = [], []
start_time = time.time()
step = 0
avg_loss = 0
valid_acc_max = 0.0
for epoch in range(1, N_EPOCHS + 1):
    print('Epoch:' + str(epoch))
    logger.info('Epoch:'+ str(epoch))
    step = train_epoch(model, optimizer, trainloader, train_loss_history, loss_func, step, epoch)
    valid_acc_max = evaluate(model, valloader, test_loss_history, loss_func, valid_acc_max, onlySaveBestAccModel)

logger.info('valid_acc_max:', valid_acc_max)
print('Execution time for training and validation:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
logger.info('Execution time for training and validation:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
