import torch
import torch.nn as nn
import math
from torch.nn.init import kaiming_normal_, constant_
from .util import predict_flow, crop_like, conv_s, conv, deconv
from torch.autograd import Variable
import sys
import time
from models.corr import corr

__all__ = ['steflow']


class FlowHead(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class ConvGRU(nn.Module):
    def __init__(self, hidden_dim=128, input_dim=192 + 128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim + input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))

        h = (1 - z) * h + z * q
        return h


class SmallUpdateBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=96):
        super(SmallUpdateBlock, self).__init__()
        # self.encoder = SmallMotionEncoder()
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=input_dim)
        # self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp):
        # motion_features = self.encoder(flow, corr)
        # inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        # delta_flow = self.flow_head(net)
        return net


class EstimationBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim=96):
        super(EstimationBlock, self).__init__()
        # self.encoder = SmallMotionEncoder()
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=input_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

    def forward(self, net, inp):
        # motion_features = self.encoder(flow, corr)
        # inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)
        return net, delta_flow


class FlowNetS_spike(nn.Module):
    expansion = 1

    def __init__(self, batchNorm=True):
        super(FlowNetS_spike, self).__init__()
        self.batchNorm = batchNorm
        self.leakyRELU = nn.LeakyReLU(0.1)

        self.num_iterative = 3

        self.conv1_2 = conv(self.batchNorm, 2, 64, kernel_size=3, stride=2)
        self.conv2_2 = conv(self.batchNorm, 64, 128, kernel_size=3, stride=2)
        self.conv3_2 = conv(self.batchNorm, 128, 256, kernel_size=3, stride=2)
        self.conv4_2 = conv(self.batchNorm, 256, 512, kernel_size=3, stride=2)

        self.md = 4
#         corr = Correlation(pad_size=self.md, kernel_size=1, max_displacement=self.md, stride1=1, stride2=1,
#                                 corr_multiply=1)
        self.nd = (2 * self.md + 1) ** 2

        self.conv1 = conv(self.batchNorm, 2, 64, kernel_size=3, stride=2)
        self.conv2 = conv(self.batchNorm, 64 + self.nd, 128, kernel_size=3, stride=2)
        self.conv3 = conv(self.batchNorm, 128 + self.nd, 256, kernel_size=3, stride=2)
        self.conv4 = conv(self.batchNorm, 256 + self.nd, 512, kernel_size=3, stride=2)

        self.conv_r00 = conv(self.batchNorm, 512 + self.nd, 512, kernel_size=3, stride=1)
        self.conv_r11 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)
        self.conv_r12 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)
        self.conv_r21 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)
        self.conv_r22 = conv(self.batchNorm, 512, 512, kernel_size=3, stride=1)

        self.deconv3 = deconv(self.batchNorm, 512 + 512 + self.nd, 128)
        self.deconv2 = deconv(self.batchNorm, 384 + 2 + self.nd, 64)
        self.deconv1 = deconv(self.batchNorm, 192 + 2 + self.nd, 4)

        self.flow_deconv4 = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1,
                                               bias=True)
        self.flow_deconv3 = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1,
                                               bias=True)
        self.flow_deconv2 = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1,
                                               bias=True)
        self.flow_deconv1 = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1,
                                               bias=True)
        self.flow_deconv0 = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1,
                                               bias=True)

        self.predict_flow4 = predict_flow(self.batchNorm, 32)
        self.predict_flow3 = predict_flow(self.batchNorm, 32)
        self.predict_flow2 = predict_flow(self.batchNorm, 32)
        self.predict_flow1 = predict_flow(self.batchNorm, 32)

        self.gru_update4 = SmallUpdateBlock(input_dim=512 + self.nd, hidden_dim=512 + self.nd)
        self.gru_update3 = SmallUpdateBlock(input_dim=256 + self.nd, hidden_dim=256 + self.nd)
        self.gru_update2 = SmallUpdateBlock(input_dim=128 + self.nd, hidden_dim=128 + self.nd)
        self.gru_update1 = SmallUpdateBlock(input_dim=64 + self.nd, hidden_dim=64 + self.nd)

        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(in_channels=512 + 512 + self.nd, out_channels=32, kernel_size=4,
                                                       stride=2, padding=1, bias=True)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(in_channels=384 + 2 + self.nd, out_channels=32, kernel_size=4,
                                                       stride=2, padding=1, bias=True)
        self.upsampled_flow2_to_1 = nn.ConvTranspose2d(in_channels=192 + 2 + self.nd, out_channels=32, kernel_size=4,
                                                       stride=2, padding=1, bias=True)
        self.upsampled_flow1_to_0 = nn.ConvTranspose2d(in_channels=68 + 2 + self.nd, out_channels=32, kernel_size=4,
                                                       stride=2, padding=1, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(3.0 / n)  # use 3 for dt1 and 2 for dt4
                m.weight.data.normal_(0, variance1)
                if m.bias is not None:
                    constant_(m.bias, 0)

            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(3.0 / n)  # use 3 for dt1 and 2 for dt4
                m.weight.data.normal_(0, variance1)
                if m.bias is not None:
                    constant_(m.bias, 0)

    def forward(self, input, image_resize, use_dt4=False, flow_seq=False):

        net1 = torch.zeros(input.size(0), 64 + self.nd, int(image_resize / 2), int(image_resize / 2)).cuda()
        net4 = torch.zeros(input.size(0), 512 + self.nd, int(image_resize / 16), int(image_resize / 16)).cuda()
        net3 = torch.zeros(input.size(0), 256 + self.nd, int(image_resize / 8), int(image_resize / 8)).cuda()
        net2 = torch.zeros(input.size(0), 128 + self.nd, int(image_resize / 4), int(image_resize / 4)).cuda()

        feature_map_0 = [0 for _ in range(input.size(4))]
        feature_map_1 = [0 for _ in range(input.size(4))]
        feature_map_2 = [0 for _ in range(input.size(4))]
        feature_map_3 = [0 for _ in range(input.size(4))]
        feature_map_4 = [0 for _ in range(input.size(4))]

        flow1_all = [torch.zeros(input.size(0), 2, int(image_resize), int(image_resize)).cuda() for _ in
                     range(input.size(4))]
        flow2_all = [torch.zeros(input.size(0), 2, int(image_resize / 2), int(image_resize / 2)).cuda() for _ in
                     range(input.size(4))]
        flow3_all = [torch.zeros(input.size(0), 2, int(image_resize / 4), int(image_resize / 4)).cuda() for _ in
                     range(input.size(4))]
        flow4_all = [torch.zeros(input.size(0), 2, int(image_resize / 8), int(image_resize / 8)).cuda() for _ in
                     range(input.size(4))]

        corr_0 = [torch.zeros(input.size(0), self.nd, int(image_resize), int(image_resize)).cuda()]
        corr_1 = [torch.zeros(input.size(0), self.nd, int(image_resize / 2), int(image_resize / 2)).cuda()]
        corr_2 = [torch.zeros(input.size(0), self.nd, int(image_resize / 4), int(image_resize / 4)).cuda()]
        corr_3 = [torch.zeros(input.size(0), self.nd, int(image_resize / 8), int(image_resize / 8)).cuda()]
        corr_4 = [torch.zeros(input.size(0), self.nd, int(image_resize / 16), int(image_resize / 16)).cuda()]

        flow_list = []
        
        for j in range(self.num_iterative):
            for i in range(input.size(4)):
                input11 = input[:, 0:2, :, :, i].cuda()
                
                current_1 = self.conv1(input11)
                feature_map_1[i] = current_1
                warp_tmp = self.flow_warp(feature_map_1[i], flow2_all[i])
                corr_1 = self.leakyRELU(corr(feature_map_1[0], warp_tmp))
                current_12 = torch.cat((current_1, corr_1), 1)
                net1 = self.gru_update1(net1, current_12)
                
                current_2 = self.conv2(net1)
                current_2_1 = self.conv2_2(current_1)
                feature_map_2[i] = current_2_1
                warp_tmp = self.flow_warp(feature_map_2[i], flow3_all[i])
                corr_2 = self.leakyRELU(corr(feature_map_2[0], warp_tmp))
                current_22 = torch.cat((current_2, corr_2), 1)
                net2 = self.gru_update2(net2, current_22)
                
                current_3 = self.conv3(net2)
                current_3_1 = self.conv3_2(current_2_1)
                feature_map_3[i] = current_3_1
                warp_tmp = self.flow_warp(feature_map_3[i], flow4_all[i])
                corr_3 = self.leakyRELU(corr(feature_map_3[0], warp_tmp))
                current_32 = torch.cat((current_3, corr_3), 1)
                net3 = self.gru_update3(net3, current_32)
                
                current_4 = self.conv4(net3)
                current_4_1 = self.conv4_2(current_3_1)
                feature_map_4[i] = current_4_1
                corr_4 = self.leakyRELU(corr(feature_map_4[0], feature_map_4[i]))
                current_42 = torch.cat((current_4, corr_4), 1)
                net4 = self.gru_update4(net4, current_42)
                
                out_rconv00 = self.conv_r00(net4)
                out_rconv11 = self.conv_r11(out_rconv00)
                out_rconv12 = self.conv_r12(out_rconv11) + out_rconv00
                out_rconv21 = self.conv_r21(out_rconv12)
                out_rconv22 = self.conv_r22(out_rconv21) + out_rconv12
                
                concat4 = torch.cat((net4, out_rconv22), 1)
                flow4 = self.predict_flow4(self.upsampled_flow4_to_3(concat4)) + flow4_all[i]
                flow4_all[i] = flow4
                # up_flow4 = self.flow_deconv4(flow4)
                out_deconv3 = self.deconv3(concat4)
                
                concat3 = torch.cat((net3, out_deconv3, flow4), 1)
                flow3 = self.predict_flow3(self.upsampled_flow3_to_2(concat3)) + flow3_all[i]
                flow3_all[i] = flow3
                # up_flow3 = self.flow_deconv3(flow3)
                out_deconv2 = self.deconv2(concat3)
                
                concat2 = torch.cat((net2, out_deconv2, flow3), 1)
                flow2 = self.predict_flow2(self.upsampled_flow2_to_1(concat2)) + flow2_all[i]
                flow2_all[i] = flow2
                # up_flow2 = self.flow_deconv2(flow2)
                out_deconv1 = self.deconv1(concat2)
                
                concat1 = torch.cat((net1, out_deconv1, flow2), 1)
                flow1 = self.predict_flow1(self.upsampled_flow1_to_0(concat1)) + flow1_all[i]
                flow1_all[i] = flow1
                
                if flow_seq and j == self.num_iterative - 1:
                    flow_list.append(flow1)
        
        if flow_seq:
#             return flow_list
            splits = input.size(4)
            split_per_dt = (int)(splits / 4)
            flow_lista = flow1_all[split_per_dt-1 : splits : split_per_dt]
            return flow_lista
        if self.training:
            if use_dt4:
                splits = input.size(4)
                split_per_dt = (int)(splits / 4)
                flow1_list = flow1_all[split_per_dt-1 : splits : split_per_dt]
                flow2_list = flow2_all[split_per_dt-1 : splits : split_per_dt]
                flow3_list = flow3_all[split_per_dt-1 : splits : split_per_dt]
                flow4_list = flow4_all[split_per_dt-1 : splits : split_per_dt]
                return flow1_list, flow2_list, flow3_list, flow4_list
            else:
                return flow1,flow2,flow3,flow4
        else:
            return flow1

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow

        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow

        """
        B, C, H, W = x.size()
        # mesh grid
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()

        if x.is_cuda:
            grid = grid.cuda()
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
        mask = nn.functional.grid_sample(mask, vgrid)

        # if W==128:
        # np.save('mask.npy', mask.cpu().data.numpy())
        # np.save('warp.npy', output.cpu().data.numpy())

        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask

    
    def flow_warp(self, x, flow12, pad='border', mode='bilinear'):
        B, _, H, W = x.size()

        base_grid = self.mesh_grid(B, H, W).type_as(x)  # B2HW

        v_grid = self.norm_grid(base_grid + flow12)  # BHW2
        im1_recons = nn.functional.grid_sample(x, v_grid, mode=mode, padding_mode=pad)
        return im1_recons
    
    
    def mesh_grid(self, B, H, W):
        # mesh grid
        x_base = torch.arange(0, W).repeat(B, H, 1)  # BHW
        y_base = torch.arange(0, H).repeat(B, W, 1).transpose(1, 2)  # BHW

        base_grid = torch.stack([x_base, y_base], 1)  # B2HW
        return base_grid


    def norm_grid(self, v_grid):
        _, _, H, W = v_grid.size()

        # scale grid to [-1,1]
        v_grid_norm = torch.zeros_like(v_grid)
        v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / (W - 1) - 1.0
        v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / (H - 1) - 1.0
        return v_grid_norm.permute(0, 2, 3, 1)  # BHW2


def steflow(data=None):
    model = FlowNetS_spike(batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model

