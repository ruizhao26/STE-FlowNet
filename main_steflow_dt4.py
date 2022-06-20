import argparse
import time
import os
import os.path as osp
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np
import models
from multiscaleloss import compute_photometric_loss, estimate_corresponding_gt_flow, flow_error_dense, smooth_loss
import datetime
from tensorboardX import SummaryWriter
from util import flow2rgb, AverageMeter, save_checkpoint
import h5py
import random
from vis_utils import *
import warnings
warnings.filterwarnings("ignore")
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__"))
parser = argparse.ArgumentParser()

parser.add_argument('--train-data-split', type=int, default=5)
parser.add_argument('--test-data-split', type=int, default=5)
parser.add_argument('--train-set', type=str, default='outdoor_day2')
parser.add_argument('--test-set', type=str, default='indoor_flying1')
parser.add_argument('-b', '--batch-size', default=8, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--gamma', '-g', type=float, default=0.7)
parser.add_argument('--print-freq', '-p', default=2000, type=int,  metavar='N', help='print frequency')
parser.add_argument('--lr', '--learning-rate', default=4e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--print-detail', '-pd', action='store_true')

parser.add_argument('--data', type=str, metavar='DIR', default='/home/datasets/mvsec', help='path to dataset')
parser.add_argument('--savedir', type=str, metavar='DATASET', default='spikeflownet', help='results save dir')
parser.add_argument('--arch', '-a', metavar='ARCH', default='steflow', choices=model_names,
                    help='model architecture, overwritten if pretrained is specified: ' + ' | '.join(model_names))
parser.add_argument('--solver', default='adam',choices=['adam','sgd'], help='solver algorithms')
parser.add_argument('-j', '--workers', default=64, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameter for adam')
parser.add_argument('--weight-decay', '--wd', default=4e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--bias-decay', default=0, type=float, metavar='B', help='bias decay')
parser.add_argument('--multiscale-weights', '-w', default=[1, 1, 1, 1], type=float, nargs=4)
parser.add_argument('--evaluate-interval', default=200, type=int, metavar='N',  help='Evaluate every \'evaluate interval\' epochs ')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=None, help='path to pre-trained model')
parser.add_argument('--no-date', action='store_true', help='don\'t append date timestamp to folder')
parser.add_argument('--div-flow', default=1, help='value by which flow will be divided. Original value is 20 but 1 with batchNorm gives good results')
parser.add_argument('--milestones', default=[5,10,20,30,40,50,70,90,110,130,150,170], metavar='N', nargs='*', help='epochs at which learning rate is divided by 2')
args = parser.parse_args()

# Initializations
best_EPE = -1
n_iter = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_resize = 256

trainenv = args.train_set
testenv = args.test_set


traindir = os.path.join(args.data, trainenv)
testdir = os.path.join(args.data, testenv)
flowgt_path = os.path.join(args.data, testenv, 'flowgt_dt4')

trainfile = traindir + '/' + trainenv + '_data.hdf5'
testfile = testdir + '/' + testenv + '_data.hdf5'

gt_file = testdir + '/' + testenv + '_gt.hdf5'


class Train_loading(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, transform=None):
        self.transform = transform
        # Training input data, label parse
        self.dt = 4
        self.split = args.train_data_split
        self.x = 260
        self.y = 346

        d_set = h5py.File(trainfile, 'r')
        self.image_raw_event_inds = np.float64(d_set['davis']['left']['image_raw_event_inds'])
        self.image_raw_ts = np.float64(d_set['davis']['left']['image_raw_ts'])
        # gray image re-size
        self.length = d_set['davis']['left']['image_raw'].shape[0]
        d_set = None

    def __getitem__(self, index):
        if index + 100 < self.length and index > 100:
            
            aa = np.zeros((self.x, self.y, int(self.dt*self.split)), dtype=np.uint8) # im_on
            bb = np.zeros((self.x, self.y, int(self.dt*self.split)), dtype=np.uint8) # im_off
                    
            for k in range(self.dt):
                im_onoff = np.load(traindir + '/count_data_sp{:02d}/'.format(self.split) + str(int(index + 1 + k)) + '.npy')
                aa[:, :, self.split*k : self.split*(k+1)] = im_onoff[0, :, :, 0:self.split]
                bb[:, :, self.split*k : self.split*(k+1)] = im_onoff[1, :, :, 0:self.split]
            
            ee0 = np.uint8(np.load(traindir + '/gray_data/' + str(int(index+0))+'.npy'))
            ee1 = np.uint8(np.load(traindir + '/gray_data/' + str(int(index+1))+'.npy'))
            ee2 = np.uint8(np.load(traindir + '/gray_data/' + str(int(index+2))+'.npy'))
            ee3 = np.uint8(np.load(traindir + '/gray_data/' + str(int(index+3))+'.npy'))
            ee4 = np.uint8(np.load(traindir + '/gray_data/' + str(int(index+4))+'.npy'))
            

            if self.transform:
                seed = np.random.randint(2147483647)

                aaa = torch.zeros(256,256,int(aa.shape[2]))
                bbb = torch.zeros(256,256,int(bb.shape[2]))
                
                sp = self.split if self.split != 13 else 5
                for p in range(int(sp*self.dt)):
                    # fix the data transformation
                    random.seed(seed)
                    torch.manual_seed(seed)
                    scale_a = aa[:, :, p].max()
                    aaa[:, :, p] = self.transform(aa[:, :, p])
                    if torch.max(aaa[:, :, p]) > 0:
                        aaa[:, :, p] = scale_a * aaa[:, :, p] / torch.max(aaa[:, :, p])

                    # fix the data transformation
                    random.seed(seed)
                    torch.manual_seed(seed)
                    scale_b = bb[:, :, p].max()
                    bbb[:, :, p] = self.transform(bb[:, :, p])
                    if torch.max(bbb[:, :, p]) > 0:
                        bbb[:, :, p] = scale_b * bbb[:, :, p] / torch.max(bbb[:, :, p])


                # fix the data transformation
                random.seed(seed)
                torch.manual_seed(seed)
                ee0 = self.transform(ee0)
                        
                # fix the data transformation
                random.seed(seed)
                torch.manual_seed(seed)
                ee1 = self.transform(ee1)
                        
                # fix the data transformation
                random.seed(seed)
                torch.manual_seed(seed)
                ee2 = self.transform(ee2)
                        
                # fix the data transformation
                random.seed(seed)
                torch.manual_seed(seed)
                ee3 = self.transform(ee3)
                        
                # fix the data transformation
                random.seed(seed)
                torch.manual_seed(seed)
                ee4 = self.transform(ee4)

            if torch.max(aaa)>0 and torch.max(bbb)>0 and torch.max(ee0)>0 and torch.max(ee1) and torch.max(ee2) and torch.max(ee3) and torch.max(ee4)>0:
                return aaa, bbb, ee0/torch.max(ee0), ee1/torch.max(ee1), ee2/torch.max(ee2), ee3/torch.max(ee3), ee4/torch.max(ee4)
            else:
                pp = torch.zeros(image_resize,image_resize,int(self.split*self.dt)) if self.split!=13 else torch.zeros(image_resize,image_resize,int(5*self.dt))
                rr = torch.zeros(1, image_resize, image_resize)
                return pp, pp, rr, rr, rr, rr, rr
        else:
            pp = torch.zeros(image_resize,image_resize,int(self.split*self.dt)) if self.split!=13 else torch.zeros(image_resize,image_resize,int(5*self.dt))
            rr = torch.zeros(1, image_resize, image_resize)
            return pp, pp, rr, rr, rr, rr, rr

    def __len__(self):
        return self.length


class Test_loading(Dataset):
    # Initialize your data, download, etc.
    def __init__(self):
        self.dt = 4
        self.xoff = 45
        self.yoff = 2
        self.split = args.test_data_split

        d_set = h5py.File(testfile, 'r')
        # Training input data, label parse
        self.image_raw_ts = np.float64(d_set['davis']['left']['image_raw_ts'])
        self.length = d_set['davis']['left']['image_raw'].shape[0]
        d_set = None

    def __getitem__(self, index):
        if (args.test_set=='outdoor_day1') and not((index>=9200 and index<9600) or (index>=10500 or index<10900)):
            pass
                        
        if (index + 20 < self.length) and (index > 20):
            
            aa = np.zeros((256, 256, int(self.dt*self.split)), dtype=np.uint8)
            bb = np.zeros((256, 256, int(self.dt*self.split)), dtype=np.uint8)
            for k in range(int(self.dt)):
                im_onoff = np.load(testdir + '/count_data_sp{:02d}/'.format(args.test_data_split) + str(int(index+1+k)) + '.npy')
                aa[:, :, self.split*k : self.split*(k+1)] = im_onoff[0, self.yoff:-self.yoff, self.xoff:-self.xoff, 0:self.split].astype(float)
                bb[:, :, self.split*k : self.split*(k+1)] = im_onoff[1, self.yoff:-self.yoff, self.xoff:-self.xoff, 0:self.split].astype(float)
                           

            return aa, bb, self.image_raw_ts[index], self.image_raw_ts[index+self.dt]
        else:
            pp = np.zeros((image_resize,image_resize,int(self.split*self.dt/2)))
            return pp, pp, np.zeros((self.image_raw_ts[index].shape)), np.zeros((self.image_raw_ts[index].shape))

    def __len__(self):
        return self.length


def train(train_loader, model, optimizer, epoch, train_writer):
    global n_iter, args
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    mini_batch_size_v = args.batch_size
    batch_size_v = 2
    
    for ww, data in enumerate(train_loader, 0):
        # get the inputs
        inputs_on, inputs_off, gray0, gray1, gray2, gray3, gray4 = data
        gray_list = [gray0, gray1, gray2, gray3, gray4]
                        
        if torch.sum(inputs_on + inputs_off) > 0:
            input_representation = torch.zeros(inputs_on.size(0), batch_size_v, image_resize, image_resize, inputs_on.size(3)).float()

            for b in range(batch_size_v):
                if b == 0:
                    input_representation[:, 0, :, :, :] = inputs_on
                elif b == 1:
                    input_representation[:, 1, :, :, :] = inputs_off

            # measure data loading time
            data_time.update(time.time() - end)

            # compute output
            input_representation = input_representation.to(device)
            output = model(input=input_representation.type(torch.cuda.FloatTensor), image_resize=image_resize, use_dt4=True)
        
            
            flow1_list, flow2_list, flow3_list, flow4_list = output
            ph_loss_list = []
            sm_loss_list = []
            
            if args.arch != 'spike_flownets_m':
                for k in range(4):
                    flow1 = flow1_list[k]
                    flow2 = flow2_list[k]
                    flow3 = flow3_list[k]
                    flow4 = flow4_list[k]
                    flow_curr_time = (flow1, flow2, flow3, flow4)

                    ph_loss_curr_time = compute_photometric_loss(gray_list[0][:, 0, :, :], gray_list[k+1][:, 0, :, :], torch.sum(input_representation, 4), flow_curr_time, weights=args.multiscale_weights)
                    ph_loss_list.append(ph_loss_curr_time)
                    sm_loss_list.append(smooth_loss(flow_curr_time))
                    
                photometric_loss = (ph_loss_list[0] + ph_loss_list[1] + ph_loss_list[2] + ph_loss_list[3]) / 4.0
                smoothness_loss = (sm_loss_list[0] + sm_loss_list[1] + sm_loss_list[2] + sm_loss_list[3]) / 4.0
                
            else:
                flows = (flow1_list, flow1_list, flow1_list, flow1_list)
                photometric_loss = compute_photometric_loss(gray_list[0][:, 0, :, :], gray_list[4][:, 0, :, :], torch.sum(input_representation, 4), flows, weights=args.multiscale_weights)
                smoothness_loss = smooth_loss(flows)

                        
            # total_loss
            loss = photometric_loss + 10 * smoothness_loss

            # compute gradient and do optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record loss and EPE
            train_writer.add_scalar('train_loss', loss.item(), n_iter)
            losses.update(loss.item(), input_representation.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if mini_batch_size_v*ww % args.print_freq < mini_batch_size_v:
                print('Epoch: [{0}][{1}/{2}]\t Time {3}\t Data {4}\t Loss {5}'
                      .format(epoch, mini_batch_size_v*ww, mini_batch_size_v*len(train_loader), batch_time, data_time, losses))
            n_iter += 1

    return losses.avg


def validate(test_loader, model, epoch, output_writers):
    global args, image_resize
    d_label = h5py.File(gt_file, 'r')
    gt_temp = np.float32(d_label['davis']['left']['flow_dist'])
    gt_ts_temp = np.float64(d_label['davis']['left']['flow_dist_ts'])
    d_label = None
    
    d_set = h5py.File(testfile, 'r')
    gray_image = d_set['davis']['left']['image_raw']

    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    batch_size_v = 2

    AEE_sum = 0.
    AEE_sum_sum = 0.
    AEE_sum_gt = 0.
    AEE_sum_sum_gt = 0.
    percent_AEE_sum = 0.
    iters = 0.
    scale = 1
                        
    print('-------------------------------------------------------')
    for i, data in enumerate(test_loader, 0):
        # -------------------------------------------------------------
        if (args.test_set == 'outdoor_day1') and not ((i >= 9200 and i <= 9600) or (i >= 10500 and i <= 10900)):
            continue
       
        inputs_on, inputs_off, st_time, ed_time = data

        if torch.sum(inputs_on + inputs_off) > 0:
        
            test_time_start = time.time()
                        
            input_representation = torch.zeros(inputs_on.size(0), batch_size_v, image_resize, image_resize, inputs_on.size(3)).float()
            
            for b in range(batch_size_v):
                if b == 0:
                    input_representation[:, 0, :, :, :] = inputs_on
                elif b == 1:
                    input_representation[:, 1, :, :, :] = inputs_off

            # compute output
            input_representation = input_representation.to(device)
            
            output = model(input_representation.type(torch.cuda.FloatTensor), image_resize)

            pred_flow = np.zeros((image_resize, image_resize, 2))
            output_temp = output.cpu()
            pred_flow[:, :, 0] = cv2.resize(np.array(output_temp[0, 0, :, :]), (image_resize, image_resize), interpolation=cv2.INTER_LINEAR)
            pred_flow[:, :, 1] = cv2.resize(np.array(output_temp[0, 1, :, :]), (image_resize, image_resize), interpolation=cv2.INTER_LINEAR)
                        
            curr_flowgt_path = os.path.join(flowgt_path, str(i) + '.npy')
            gt_flow = np.load(curr_flowgt_path)
            

            image_size = pred_flow.shape
            full_size = gt_flow.shape
            xsize = full_size[1]
            ysize = full_size[0]
            xcrop = image_size[1]
            ycrop = image_size[0]
            xoff = (xsize - xcrop) // 2
            yoff = (ysize - ycrop) // 2

            gt_flow = gt_flow[yoff:-yoff, xoff:-xoff, :]

            AEE, percent_AEE, n_points, AEE_sum_temp, AEE_gt, AEE_sum_temp_gt = flow_error_dense(gt_flow, pred_flow, (torch.sum(torch.sum(torch.sum(input_representation, dim=0), dim=0), dim=2)).cpu(), is_car=(args.test_set=='outdoor_day1'))

            AEE_sum = AEE_sum + args.div_flow * AEE
            AEE_sum_sum = AEE_sum_sum + AEE_sum_temp

            AEE_sum_gt = AEE_sum_gt + args.div_flow * AEE_gt
            AEE_sum_sum_gt = AEE_sum_sum_gt + AEE_sum_temp_gt

            percent_AEE_sum += percent_AEE

             # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            iters += 1

            test_time = time.time() - test_time_start

            istr = ('{:05d} / {:05d}  AEE: {:2.6f}  meanAEE:{:2.6f}  test_time:{:.1f}'.format(i, len(test_loader), AEE,
                                                                                          AEE_sum / iters, test_time))
            if args.print_detail:
                print(istr)

    print('-------------------------------------------------------')
    print('Mean AEE: {:.6f}, sum AEE: {:.6f}, Mean AEE_gt: {:.6f}, sum AEE_gt: {:.6f}, mean %AEE: {:.6f}, 1 - mean %AEE: {:.6f}, # pts: {:.6f}'
                  .format(AEE_sum / iters, AEE_sum_sum / iters, AEE_sum_gt / iters, AEE_sum_sum_gt / iters, percent_AEE_sum / iters, 1.-percent_AEE_sum / iters, n_points))
    print('-------------------------------------------------------')
    gt_temp = None

    return AEE_sum / iters


def main():
    global args, best_EPE, image_resize, device
    timestamp1 = datetime.datetime.now().strftime("%m-%d")
    timestamp2 = datetime.datetime.now().strftime("%H%M%S")
    
    save_folder_name = '{},{},{},b{},lr{},{}'.format(
        args.arch,
        args.solver,
        args.epochs,
        args.batch_size,
        args.lr,
        timestamp2)
    save_root = os.path.join(args.savedir, timestamp1)
    save_path = os.path.join(save_root, save_folder_name)

    if not args.evaluate:
        print('=> Everything will be saved to {}'.format(save_path))
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    train_writer = SummaryWriter(os.path.join(save_path,'train'))
    test_writer = SummaryWriter(os.path.join(save_path,'test'))
    output_writers = []
    for i in range(3):
        output_writers.append(SummaryWriter(os.path.join(save_path,'test',str(i))))

    # Data loading code
    co_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop((256, 256), scale=(0.5, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
        transforms.ToTensor(),
    ])
    Test_dataset = Test_loading()
    test_loader = DataLoader(dataset=Test_dataset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=args.workers)

    # create model
    if args.pretrained:
        network_data = torch.load(args.pretrained)
        #args.arch = network_data['arch']
        print("=> using pre-trained model '{}'".format(args.arch))
    else:
        network_data = None
        print("=> creating model '{}'".format(args.arch))

    model = models.__dict__[args.arch](network_data).cuda()
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    assert(args.solver in ['adam', 'sgd'])
    print('=> setting {} solver'.format(args.solver))
    param_groups = [{'params': model.module.bias_parameters(), 'weight_decay': args.bias_decay},
                    {'params': model.module.weight_parameters(), 'weight_decay': args.weight_decay}]
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(param_groups, args.lr, betas=(args.momentum, args.beta))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(param_groups, args.lr, momentum=args.momentum)

    if args.evaluate:
        with torch.no_grad():
            best_EPE = validate(test_loader, model, -1, output_writers)
        return

    Train_dataset = Train_loading(transform=co_transform)
    train_loader = DataLoader(dataset=Train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.workers)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.7)

    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()

        # train for one epoch
        train_loss = train(train_loader, model, optimizer, epoch, train_writer)
        train_writer.add_scalar('mean loss', train_loss, epoch)

        filename = 'epoch{:02d}_sp{:02d}_dt4_ckpt.pth.tar'.format(epoch + 1, args.train_data_split)
        is_best = False
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.module.state_dict(),
            'best_EPE': best_EPE,
            'div_flow': args.div_flow
        }, is_best, save_path, filename=filename)
        
        # Test at every 5 epoch during training
        if (epoch + 1)%args.evaluate_interval == 0:
            # evaluate on validation set
            with torch.no_grad():
                EPE = validate(test_loader, model, epoch, output_writers)
            test_writer.add_scalar('mean EPE', EPE, epoch)

if __name__ == '__main__':
    main()
