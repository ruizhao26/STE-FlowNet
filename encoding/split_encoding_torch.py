import numpy as np
import os
import h5py
import argparse
import time
import torch
import torch.nn.functional as F


class Events(object):
    def __init__(self, num_events, width=346, height=260):
        self.data = np.rec.array(None, dtype=[('x', np.uint16), ('y', np.uint16), ('p', np.bool_), ('ts', np.float64)],
                                 shape=(num_events))
        self.width = width
        self.height = height

    def h5_to_torch(self, e):
        return torch.stack((
        torch.from_numpy(e[:,0]),
        torch.from_numpy(e[:,1]),
        torch.from_numpy(e[:,2]),
        torch.from_numpy(e[:,3]), 
        ), dim=-1)

    def split_events(self, events):
        return events[:, 0].long(), events[:, 1].long(), events[:, 2].float(), events[:, 3].float()

    def events_representation_single_polar(self, events):
        """
        events(N x 4): raw events (single polar)
        xs(N x 1), ys(N x 1), ts(N x 1), ps(N x 1): Events attributes splits to be 4 list
        tavg(scalar): average of normalized time stamp for each pixel
        tstd(scalar): standard deviation of normalizaed time stamp for each pixel
        repre_img(H x W): the representation image
        gaussian_part1(scalar): the $1/(sqrt(2*pi)*sigma)$
        gaussian_part2(N x 1): the exponent in gaussian distribution of the Exp
        gaussian_value_at_event(N x 1): the sampled gaussian value for each event
        """
        
        ys, xs, ts, ps = self.split_events(events)

        repre_img = torch.zeros((self.height, self.width)).float()
        count_img = torch.zeros((self.height, self.width)).float()
        if not args.use_no_cuda:
            repre_img = repre_img.cuda()
            count_img = count_img.cuda()

        t_avg = ts.mean()
        t_std = ts.std() + 0.0001 if ts.shape[0] > 1 else 1

        gaussian_part1 = 1 / np.sqrt(2*np.pi) / t_std
        gaussian_part2 = - (ts - t_avg)**2 / (2 * t_std**2)
        gaussian_value_at_event = gaussian_part1 * torch.exp(gaussian_part2)

        # normalizing factor
        lam = ps.abs().sum() / gaussian_value_at_event.sum()
        repre_img.index_put_((xs, ys), torch.ceil(gaussian_value_at_event * lam), accumulate=True)
        
        return repre_img.int()

    def generate_fimage(self, input_event=0, gray=0, image_raw_event_inds_temp=0, image_raw_ts_temp=0, dt_time_temp=0):
        print(image_raw_event_inds_temp.shape, image_raw_ts_temp.shape)

        split_interval = image_raw_ts_temp.shape[0]
        if (args.save_env == 'outdoor_day1' or args.save_env == 'outdoor_day2') and args.data_split == 5:
            data_split = 13
        
        t_index = 0
        encoding_length = split_interval - (dt_time_temp - 1)

        for i in range(split_interval - (dt_time_temp - 1)):
            
            td_img_c = torch.zeros((2, self.height, self.width, data_split), dtype=torch.uint8)
            if not args.use_no_cuda:
                td_img_c = td_img_c.cuda()
            if image_raw_event_inds_temp[i - 1] < 0:
                frame_data = input_event[0:image_raw_event_inds_temp[i + (dt_time_temp - 1)], :]
            else:
                frame_data = input_event[
                             image_raw_event_inds_temp[i - 1]:image_raw_event_inds_temp[i + (dt_time_temp - 1)], :]
            
            st = time.time()
            if frame_data.size > 0:
                # Channel Explaination
                # Channel 0: Gaussian Weight Count for On Spikes
                # Channel 1: Gaussian Weight Count for Off Spikes
                slice_len = int(frame_data.shape[0] / data_split)
                for m in range(data_split):
                    slice_frame_data = frame_data[slice_len * m : slice_len * (m+1), :]
                    # Converting slice_frame_data to PyTorch
                    slice_frame_data = self.h5_to_torch(slice_frame_data)
                    if not args.use_no_cuda:
                        slice_frame_data = slice_frame_data.cuda()

                    # Firstly, Normalization in Time to (0, 1)
                    slice_frame_data[:, 2] = (slice_frame_data[:, 2] - torch.min(slice_frame_data[:, 2])) / (torch.max(slice_frame_data[:, 2]) - torch.min(slice_frame_data[:, 2]))

                    pos_event = slice_frame_data[slice_frame_data[:, 3] == 1]
                    neg_event = slice_frame_data[slice_frame_data[:, 3] == -1]

                    pos_repre = self.events_representation_single_polar(pos_event)
                    neg_repre = self.events_representation_single_polar(neg_event)

                    td_img_c[0, :, :, m] = pos_repre
                    td_img_c[1, :, :, m] = neg_repre

            t_index = t_index + 1
            
            if not args.use_no_cuda:
                td_img_c = td_img_c.cpu().numpy()
            else:
                td_img_c = td_img_c.numpy()

            if (args.save_env == 'outdoor_day1' or args.save_env == 'outdoor_day2') and args.data_split == 5:
                td_img_c = td_img_c[:, :, :, 0:13:3]
                
                
            np.save(os.path.join(count_dir, str(i)), td_img_c)
            np.save(os.path.join(gray_dir, str(i)), gray[i, :, :])

            if args.sparse_print:
                if i % 1000 == 0:
                    print('Dataset {:s} sp={:02d} Finish Encoding {:05d} / {:05d} Time: {:.2f} !'.format(args.save_env, args.data_split, i, encoding_length - 1, time.time()-st))
            else:
                print('Dataset {:s} sp={:02d} Finish Encoding {:05d} / {:05d} Time: {:.2f} !'.format(args.save_env, args.data_split, i, encoding_length - 1, time.time()-st))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spike Encoding')
    parser.add_argument('--data-split', '-sp', type=int, default=5)
    parser.add_argument('--save-dir', '-sd', type=str, default='/home/datasets/mvsec', metavar='PARAMS',
                        help='Main Directory to save all encoding results')
    parser.add_argument('--save-env', '-se', type=str, default='indoor_flying1', metavar='PARAMS',
                        help='Sub-Directory name to save Environment specific encoding results')
    parser.add_argument('--use_no_cuda', '-nc', action='store_true', help='Use cuda to encode')
    parser.add_argument('--old', '-o', action='store_true', help='Encoding old version: Events Count')
    parser.add_argument('--sparse_print', '-s', action='store_true', help='saprse print log')
    args = parser.parse_args()
    os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

    args.data_path = args.save_dir + '/' + args.save_env + '/' + args.save_env + '_data.hdf5'

    #### Check if CUDA is available
    if not torch.cuda.is_available() and not args.use_no_cuda:
        args.use_no_cuda = True
        print('No GPU is available, the encoding will implment on CPU')

    #### Data Path
    save_path = os.path.join(args.save_dir, args.save_env)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    #### Save Direction
    count_dir = os.path.join(save_path, 'count_data_sp{:02d}'.format(args.data_split))
    if not os.path.exists(count_dir):
        print('making', count_dir)
        os.makedirs(count_dir)

    gray_dir = os.path.join(save_path, 'gray_data'.format(args.data_split))
    if not os.path.exists(gray_dir):
        os.makedirs(gray_dir)


    #### Read Files
    d_set = h5py.File(args.data_path, 'r')
    raw_data = d_set['davis']['left']['events']
    image_raw_event_inds = d_set['davis']['left']['image_raw_event_inds']
    image_raw_ts = np.float64(d_set['davis']['left']['image_raw_ts'])
    gray_image = d_set['davis']['left']['image_raw']
    d_set = None

    dt_time = 1

    td = Events(raw_data.shape[0])
    # Events
    td.generate_fimage(input_event=raw_data, gray=gray_image, image_raw_event_inds_temp=image_raw_event_inds,
                    image_raw_ts_temp=image_raw_ts, dt_time_temp=dt_time)
    raw_data = None

    print('Encoding complete!')
