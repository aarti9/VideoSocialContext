import cv2
import logging
import numpy as np
import os
import time
import torch
import torch.nn.functional as F
import pickle
from sklearn.metrics import accuracy_score
from functools import partial
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 2
num_frames = 16
N_EPOCHS = 20
model_name = "R2Plus1D"
training_set = "Tset" # Tset or TVset
onlySaveBestAccModel = True

run_identity = "vgaf_"+ model_name +str(num_frames)+"F_train_"+training_set
print("******************************************")
print("Experimental run for : " + run_identity)
if onlySaveBestAccModel:
    print("Note: Only best accuracy checkpoint in each epoch will be saved")
else:
    print("Note: checkpoint in each epoch will be saved")
print("******************************************")

if training_set == "TVset":
    train_videos_path = 'TrainValAll'
    val_videos_path = 'TestAll'
    train_videos_frames_path = '/scratch/aarti9/TrainVal_'+str(num_frames)+'_Frames'
    val_videos_frames_path = '/scratch/aarti9/Test_'+str(num_frames)+'_Frames'
else:
    train_videos_path = 'TrainAll'
    val_videos_path = 'ValAll'
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

def get_frames(filename, n_frames=1):
    frames = []
    v_cap = cv2.VideoCapture(filename)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_list = np.linspace(0, v_len - 1, n_frames + 1, dtype=np.int16)
    frame_dims = np.array([224, 224, 3])
    for fn in range(v_len):
        success, frame = v_cap.read()
        if success is False:
            continue
        if (fn in frame_list):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (frame_dims[0], frame_dims[1]))
            frames.append(frame)
    v_cap.release()
    return frames, v_len

def get_inplanes():
    return [64, 128, 256, 512]
def conv1x3x3(in_planes, mid_planes, stride=1):
    return nn.Conv3d(in_planes,
                     mid_planes,
                     kernel_size=(1, 3, 3),
                     stride=(1, stride, stride),
                     padding=(0, 1, 1),
                     bias=False)
def conv3x1x1(mid_planes, planes, stride=1):
    return nn.Conv3d(mid_planes,
                     planes,
                     kernel_size=(3, 1, 1),
                     stride=(stride, 1, 1),
                     padding=(1, 0, 0),
                     bias=False)
def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        n_3d_parameters1 = in_planes * planes * 3 * 3 * 3
        n_2p1d_parameters1 = in_planes * 3 * 3 + 3 * planes
        mid_planes1 = n_3d_parameters1 // n_2p1d_parameters1
        self.conv1_s = conv1x3x3(in_planes, mid_planes1, stride)
        self.bn1_s = nn.BatchNorm3d(mid_planes1)
        self.conv1_t = conv3x1x1(mid_planes1, planes, stride)
        self.bn1_t = nn.BatchNorm3d(planes)
        n_3d_parameters2 = planes * planes * 3 * 3 * 3
        n_2p1d_parameters2 = planes * 3 * 3 + 3 * planes
        mid_planes2 = n_3d_parameters2 // n_2p1d_parameters2
        self.conv2_s = conv1x3x3(planes, mid_planes2)
        self.bn2_s = nn.BatchNorm3d(mid_planes2)
        self.conv2_t = conv3x1x1(mid_planes2, planes)
        self.bn2_t = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1_s(x)
        out = self.bn1_s(out)
        out = self.relu(out)
        out = self.conv1_t(out)
        out = self.bn1_t(out)
        out = self.relu(out)
        out = self.conv2_s(out)
        out = self.bn2_s(out)
        out = self.relu(out)
        out = self.conv2_t(out)
        out = self.bn2_t(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        n_3d_parameters = planes * planes * 3 * 3 * 3
        n_2p1d_parameters = planes * 3 * 3 + 3 * planes
        mid_planes = n_3d_parameters // n_2p1d_parameters
        self.conv2_s = conv1x3x3(planes, mid_planes, stride)
        self.bn2_s = nn.BatchNorm3d(mid_planes)
        self.conv2_t = conv3x1x1(mid_planes, planes, stride)
        self.bn2_t = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2_s(out)
        out = self.bn2_s(out)
        out = self.relu(out)
        out = self.conv2_t(out)
        out = self.bn2_t(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        n_3d_parameters = 3 * self.in_planes * conv1_t_size * 7 * 7
        n_2p1d_parameters = 3 * 7 * 7 + conv1_t_size * self.in_planes
        mid_planes = n_3d_parameters // n_2p1d_parameters
        self.conv1_s = nn.Conv3d(n_input_channels,
                                 mid_planes,
                                 kernel_size=(1, 7, 7),
                                 stride=(1, 2, 2),
                                 padding=(0, 3, 3),
                                 bias=False)
        self.bn1_s = nn.BatchNorm3d(mid_planes)
        self.conv1_t = nn.Conv3d(mid_planes,
                                 self.in_planes,
                                 kernel_size=(conv1_t_size, 1, 1),
                                 stride=(conv1_t_stride, 1, 1),
                                 padding=(conv1_t_size // 2, 0, 0),
                                 bias=False)
        self.bn1_t = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()
        out = torch.cat([out.data, zero_pads], dim=1)
        return out
    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))
        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1_s(x)
        x = self.bn1_s(x)
        x = self.relu(x)
        x = self.conv1_t(x)
        x = self.bn1_t(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]
    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)
    return model.cuda(0)

model = generate_model(50, shortcut_type='B',n_classes=700)
pretrain = torch.load("r2p1d50_K_200ep.pth")
model.load_state_dict(pretrain['state_dict'], strict=False)

count = 0
for child in model.children():
    count+=1
print('no of resnet layers',count)

# output_classes = 3
count = 0
for child in model.children():
    count+=1
    if count < 4:
        for param in child.parameters():
            param.requires_grad = False
    # for param in child.parameters():
    #     print(count, param.requires_grad)

num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)          
# model.fc = nn.Linear(num_ftrs, output_classes)

model = model.to(device)
# print(model)
# print(summary(model,(3,32,112,112)))

cce_loss = nn.CrossEntropyLoss()
# specify loss function
optimizer = torch.optim.SGD(model.parameters(), lr= 0.0002,
                                         momentum=0.9,
                                         weight_decay=1e-9)

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

def train(model, step, num_epochs, valid_acc_max, onlySaveBestAccModel):
    train_on_gpu = torch.cuda.is_available()

    total_samples = len(trainloader.dataset)
    valid_loss_min = np.Inf # track change in validation loss
    # model = torch.load('drive/MyDrive/affect_test_videos/saveWeights/model_vgaf3_3F_r3d50_KM_200ep_till_3epoch.pt')
    
    for epoch in range(1, num_epochs+1):
        print('Epoch:', epoch)
        logger.info('Epoch:', epoch)
        epoch_start_time = time.time()
        top1 = AverageMeter('acc@1', ':6.2f')
        top5 = AverageMeter('acc@5', ':6.2f')
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        train_acc = 0.0
        val_acc = 0.0
        train_y_true = []
        train_y_pred = []
        val_y_true = []
        val_y_pred = []
        ###################
        # train the model #
        ###################
        model.train()
        train_loader = tqdm(trainloader, desc='Loading train data')
        avg_loss = 0
        i = 0
        for data_1, target in trainloader:
            i = i + 1
            # move tensors to GPU if CUDA is available
            data_1, target = Variable(data_1), Variable(target)
            target = torch.tensor(target, dtype=torch.long).cuda(0)
            # target = target.clone().detach()
            data_1 = data_1.to(device)
            target = target.to(device)
            data_1 = data_1.permute(0,4,1,2,3)
           # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            pred = model(data_1.float())
            loss = cce_loss(pred, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            prec1, prec5 = accuracy(pred, target, topk=(1, 5))
            top1.update(prec1.item(), pred.size(0))
            top5.update(prec5.item(), pred.size(0))

            step += 1
            avg_loss += loss.item()
            if i % 5 == 0:
                train_loader.set_description(
                    'train_loss: %.3f, acc1: %.3f, acc5: %.3f, step %d' %
                    (avg_loss / (i + 1), top1.avg, top5.avg, step))

                train_loss_history.append(loss.item())

            train_loss += loss.item()*data_1.size(0) #loss.data[0]*images.size(0)
            (max_vals, arg_maxs) = torch.max(pred, dim=1)
            train_y_pred.extend(arg_maxs.cpu().detach().numpy())
            train_y_true.extend(target.cpu().detach().numpy())
        ######################    
        # validate the model #
        ######################
        model.eval()
        top1 = AverageMeter('acc@1', ':6.2f')
        top5 = AverageMeter('acc@5', ':6.2f')
        total_samples = len(valloader.dataset)
        val_loader = tqdm(valloader, desc='\r')

        # correct_samples = 0
        total_loss = 0
        with torch.no_grad():
            for  data_1, target in valloader:
                data_1, target = Variable(data_1), Variable(target)
                # move tensors to GPU if CUDA is available
                target = torch.tensor(target, dtype=torch.long).cuda(0)
                # target = target.clone().detach()
                data_1 = data_1.to(device)
                target = target.to(device)
                data_1 = data_1.permute(0,4,1,2,3)
                # forward pass: compute predicted outputs by passing inputs to the model
                pred = model(data_1.float())
                # calculate the batch loss
                loss = cce_loss(pred, target)
                # update average validation loss
                prec1, prec5 = accuracy(pred, target, topk=(1, 5))
                top1.update(prec1.item(), pred.size(0))
                top5.update(prec5.item(), pred.size(0))

                total_loss += loss.item()
                val_loader.set_description(
                    'val_loss:   %.3f, acc1: %.3f, acc5: %.3f' %
                    (total_loss / (i + 1), top1.avg, top5.avg))
                # correct_samples += pred.eq(target).sum()

                valid_loss += loss.item() * data_1.size(0)
                (max_vals, arg_maxs) = torch.max(pred, dim=1)
                val_y_pred.extend(arg_maxs.cpu().detach().numpy())
                val_y_true.extend(target.cpu().detach().numpy())

            test_loss_history.append(total_loss / len(valloader))
            print('val_loss:   %.3f, acc1: %.3f, acc5: %.3f' % (total_loss / len(valloader), top1.avg, top5.avg))
            logger.info('val_loss:   %.3f, acc1: %.3f, acc5: %.3f' % (total_loss / len(valloader), top1.avg, top5.avg))

        # calculate average losses
        val_y_pred = np.asarray(val_y_pred)
        val_y_true = np.asarray(val_y_true)
        train_y_pred = np.asarray(train_y_pred)
        train_y_true = np.asarray(train_y_true)
        train_loss = train_loss/len(trainloader.sampler)#(train_mini_batch_counter-1)
        valid_loss = valid_loss/len(valloader.sampler)#(val_mini_batch_counter-1)
        train_acc = accuracy_score(train_y_true, train_y_pred)
        val_acc = accuracy_score(val_y_true, val_y_pred)
        # print training/validation statistics 
        print('Epoch: {} \tTrain Loss: {:.6f} \tVal Loss: {:.6f} \tTrain acc:{:.5f} \tVal acc:{:.5f} \tTime:{:.2f}'.format(
            epoch, train_loss, valid_loss,train_acc,val_acc,time.time() - epoch_start_time))
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

    return step, valid_acc_max

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
step, valid_acc_max = train(model, step, N_EPOCHS, valid_acc_max, onlySaveBestAccModel)

logger.info('valid_acc_max:', valid_acc_max)
print('Execution time for training and validation:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
logger.info('Execution time for training and validation:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
