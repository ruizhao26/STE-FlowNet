import argparse
import time
import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import datetime
import h5py
import random
from util import flow2rgb, AverageMeter, save_checkpoint
from multiscaleloss import compute_photometric_loss, estimate_corresponding_gt_flow, flow_error_dense, smooth_loss
from vis_utils import *
import warnings

warnings.filterwarnings("ignore")
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, metavar='DIR', default='/home/datasets/mvsec', help='path to dataset')
parser.add_argument('--test-set', '-ts', type=str, default='indoor_flying1')
parser.add_argument('--dt', '-dt', type=int, default=1)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')
args = parser.parse_args()


# Initializations
image_resize = 256
trainenv = 'outdoor_day2'
testenv = args.test_set
traindir = osp.join(args.data, trainenv)
testdir = osp.join(args.data, testenv)
trainfile = traindir + '/' + trainenv + '_data.hdf5'
testfile = testdir + '/' + testenv + '_data.hdf5'
gt_file = testdir + '/' + testenv + '_gt.hdf5'


class Test_loading_dt1(Dataset):
    # Initialize your data, download, etc.
    def __init__(self):
        self.dt = 1
        self.xoff = 45
        self.yoff = 2
        self.split = 5

        d_set = h5py.File(testfile, 'r')

        # Training input data, label parse
        self.image_raw_ts = np.float64(d_set['davis']['left']['image_raw_ts'])
        self.length = d_set['davis']['left']['image_raw'].shape[0]
        d_set = None

    def __getitem__(self, index):
        if (index + 20 < self.length) and (index > 20):
            aa = np.zeros((256, 256, self.split), dtype=np.uint8)
            bb = np.zeros((256, 256, self.split), dtype=np.uint8)

            im_onoff = np.load(testdir + '/count_data_sp05/' + str(int(index + 1)) + '.npy')
            aa[:, :, :] = im_onoff[0, self.yoff:-self.yoff, self.xoff:-self.xoff, 0:self.split].astype(float)
            bb[:, :, :] = im_onoff[1, self.yoff:-self.yoff, self.xoff:-self.xoff, 0:self.split].astype(float)
            
            return aa, bb, self.image_raw_ts[index], self.image_raw_ts[index + self.dt]
        else:
            pp = np.zeros((image_resize, image_resize, self.split))
            return pp, pp, np.zeros((self.image_raw_ts[index].shape)), np.zeros(
                (self.image_raw_ts[index].shape))

    def __len__(self):
        return self.length


class Test_loading_dt4(Dataset):
    # Initialize your data, download, etc.
    def __init__(self):
        self.dt = 4
        self.xoff = 45
        self.yoff = 2
        self.split = args.data_split

        d_set = h5py.File(testfile, 'r')
        # Training input data, label parse
        self.image_raw_ts = np.float64(d_set['davis']['left']['image_raw_ts'])
        self.length = d_set['davis']['left']['image_raw'].shape[0]
        d_set = None

    def __getitem__(self, index):
        if (index + 20 < self.length) and (index > 20):
            aa = np.zeros((256, 256, int(self.dt*self.split)), dtype=np.uint8)
            bb = np.zeros((256, 256, int(self.dt*self.split)), dtype=np.uint8)

            for k in range(int(self.dt)):
                im_onoff = np.load(testdir + '/count_data_sp05/' + str(int(index + 1 + k)) + '.npy')
                aa[:,:,self.split*k:self.split*(k+1)] = im_onoff[0,self.yoff:-self.yoff,self.xoff:-self.xoff,:].astype(float)
                bb[:,:,self.split*k:self.split*(k+1)] = im_onoff[1,self.yoff:-self.yoff,self.xoff:-self.xoff,:].astype(float)
            return aa, bb, self.image_raw_ts[index], self.image_raw_ts[index+self.dt]
        else:
            pp = np.zeros((image_resize,image_resize,int(self.split*self.dt/2)))
            return pp, pp, np.zeros((self.image_raw_ts[index].shape)), np.zeros((self.image_raw_ts[index].shape))

    def __len__(self):
        return self.length


def generate_flowgt(test_loader):
    global args, image_resize, sp_threshold
    d_label = h5py.File(gt_file, 'r')
    gt_temp = np.float32(d_label['davis']['left']['flow_dist'])
    gt_ts_temp = np.float64(d_label['davis']['left']['flow_dist_ts'])
    d_label = None
    
    d_set = h5py.File(testfile, 'r')
    gray_image = d_set['davis']['left']['image_raw']


    for i, data in enumerate(test_loader, 0):

        if osp.exists(osp.join(flowgt_path, str(i)+'.npy')):
            print('flowgt_dt{:d} of {:s} frame {:05d} already exists'.format(args.dt, args.test_set, i))
            continue
            
        # -------------------------------------------------------------
        if (args.test_set == 'outdoor_day1') and not ((i>=9200 and i<=9600) or (i>=10500 and i<=10900)):
            continue
        # -------------------------------------------------------------
        inputs_on, inputs_off, st_time, ed_time = data
        
        if torch.sum(inputs_on + inputs_off) > 0:

            time_start = time.time()

            U_gt_all = np.array(gt_temp[:, 0, :, :])
            V_gt_all = np.array(gt_temp[:, 1, :, :])

            U_gt, V_gt = estimate_corresponding_gt_flow(U_gt_all, V_gt_all, gt_ts_temp, np.array(st_time), np.array(ed_time))
            gt_flow = np.stack((U_gt, V_gt), axis=2)

            curr_path = osp.join(flowgt_path, str(i))
            np.save(curr_path, gt_flow)
            print('Finish saving flowgt_dt{:d} of {:s} frame {:05d} time={:.2f} '.format(args.dt, args.test_set, i, time.time() - time_start))

    return


def main():
    global args, image_resize

    flowgt_path = osp.join(args.data, args.test_set, 'flowgt_dt{:d}'.format(args.dt))
    if not osp.exists(flowgt_path):
        os.makedirs(flowgt_path)


    if args.dt == 1:
        Test_dataset = Test_loading_dt1()
    elif args.dt == 4:
        Test_dataset = Test_loading_dt4()

    test_loader = DataLoader(dataset=Test_dataset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=args.workers)

    generate_flowgt(test_loader)


if __name__ == '__main__':
    main()
