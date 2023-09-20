import math

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from transformer_pos_enc import get_3d_sincos_pos_embed_from_point
from transformer import TransformerBlock as TransformerBlock


class transformer_net(nn.Module):
    def __init__(self, encoder_dim=64):
        super(transformer_net, self).__init__()
        self.encoder_dim = encoder_dim
        self.tl1 = TransformerBlock(1, encoder_dim // 2, k=7)
        self.bn1 = nn.BatchNorm1d(encoder_dim // 2)

        self.tl2 = TransformerBlock(encoder_dim // 2, encoder_dim, k=7)
        self.bn2 = nn.BatchNorm1d(encoder_dim)

        self.tl3 = TransformerBlock(encoder_dim, encoder_dim, k=7)
        self.bn3 = nn.BatchNorm1d(encoder_dim)

        self.tl4 = TransformerBlock(encoder_dim, encoder_dim, k=7)
        self.bn4 = nn.BatchNorm1d(encoder_dim)

        self.cls = nn.Sequential(nn.Conv1d(self.encoder_dim, self.encoder_dim, 1),
                                 nn.BatchNorm1d(self.encoder_dim),
                                 nn.ReLU(),
                                 nn.Conv1d(self.encoder_dim, 2, 1))

    def forward(self, data):
        data = data[:, 0, :, :, :].unsqueeze(1)
        B = data.shape[0]
        data_width = data.shape[-1]
        xyz = torch.stack(torch.meshgrid([torch.arange(0, data_width, dtype=data.dtype, device=data.device)]*3))
        xyz = (xyz.reshape(3, -1).permute(1,0)).unsqueeze(0).repeat(B, 1, 1)
        data = data.reshape(B, 1, -1).permute(0, 2, 1)
        # B, C, D, _, _ = data.shape
        # data = data.permute(0,2,3,4,1).reshape(-1)
        # data = get_1d_sincos_pos_embed_from_grid(20, data)
        # data = data.reshape(B, D, D, D, -1).permute(0,4,1,2,3)
        f1 = F.relu(self.bn1(self.tl1(data, xyz)[0].permute(0, 2, 1)).permute(0, 2, 1))
        f2 = F.relu(self.bn2(self.tl2(f1, xyz)[0].permute(0, 2, 1)).permute(0, 2, 1))

        f3 = F.relu(self.bn3(self.tl3(f2, xyz)[0].permute(0, 2, 1)).permute(0, 2, 1))
        f4 = F.relu(self.bn4(self.tl4(f3, xyz)[0].permute(0, 2, 1)).permute(0, 2, 1))

        #
        # f7 = F.relu(self.ln7(self.conv7(F.pad(f6, (1, 1, 1, 1, 1, 1), 'constant', 0.0))))
        # f8 = F.relu(self.ln8(self.conv8(F.pad(f7, (1, 1, 1, 1, 1, 1), 'constant', 0.0))))

        feature = f4.squeeze()
        feature = feature.permute(0,2,1).reshape(B, -1, data_width, data_width, data_width)
        feature = feature[:, :, data_width // 2 - 1:data_width // 2 + 1,
                  data_width // 2 - 1:data_width // 2 + 1, data_width // 2 - 1:data_width // 2 + 1]
        feature = feature.reshape(B, -1, 8)  # .permute(0,2,1)
        out = self.cls(feature)
        return out

    def cal_loss(self, out, label):
        label = label.long()
        label = torch.clamp(label + 1, max=1)

        B, N = out.shape[0], out.shape[2]
        out = out.permute(0, 2, 1).reshape(-1, 2)
        logit = F.log_softmax(out, dim=-1)
        label = label.long().reshape(-1)
        # label = ((label[:,1:] + label[:, 0].unsqueeze(1))%2).reshape(-1)
        loss = F.nll_loss(logit, label, reduction='none').reshape(B, N).mean(-1)
        loss_sys = F.nll_loss(logit, (label + 1) % 2, reduction='none').reshape(B, N).mean(-1)

        # pred_label = out.argmax(dim=1).reshape(B, N)
        # label = label.reshape(B, N)
        # acc = (label == pred_label).float().mean()
        acc_head = out.argmax(dim=1).reshape(B, N)
        acc_tail = out.argmin(dim=1).reshape(B, N)

        acc_head = (acc_head == label.reshape(B, N)).float().mean(-1)
        acc_tail = (acc_tail == label.reshape(B, N)).float().mean(-1)
        acc = (torch.stack([acc_head, acc_tail]).max(0)[0] + 1e-3).floor().mean()

        loss_com = torch.stack([loss_sys, loss]).min(0)[0].mean()
        return loss_com, acc

    def predict(self, out, label):
        B, _, N = out.shape
        out = out.permute(0, 2, 1).reshape(-1, 2)
        logit = F.log_softmax(out, dim=-1)
        label = label.reshape(-1).long()
        label = torch.clamp(label + 1, max=1)
        confidence = (F.softmax(out, -1)[:, 0] - 0.5).abs()
        confidence = confidence.reshape(B, N).min(-1)[0]
        pred = logit.argmax(dim=1).reshape(B, N)
        sign = (pred - 0.5) * 2
        sign[torch.where(confidence < 0.1)] = 1

        acc_head = logit.argmax(dim=1).reshape(B, N)
        acc_tail = logit.argmin(dim=1).reshape(B, N)
        label = label.reshape(B, N)
        acc_head = (acc_head == label).float().mean(-1)
        acc_tail = (acc_tail == label).float().mean(-1)
        acc = (torch.stack([acc_head, acc_tail]).max(0)[0] + 1e-3).floor().mean()
        return sign, acc

class dual_conv(nn.Module):
    def __init__(self, input_dim, out_dim, kernel_size, stride=1, padding=1):
        super(dual_conv, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.stride = stride
        self.padding = padding
        self.g_weight = nn.Parameter(torch.Tensor(out_dim, input_dim, 4))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.kaiming_uniform_(self.g_weight, a=math.sqrt(5))

    def forward(self, data):
        weight_index = [2, 1, 2, 1, 0, 1, 2, 1, 2, 1, 0, 1, 0, 3, 0, 1, 0, 1, 2, 1, 2, 1, 0, 1, 2, 1, 2]
        weight = self.g_weight[:, :, weight_index].reshape(self.out_dim, self.input_dim, 3, 3, 3)
        out = F.conv3d(data, weight, self.bias, stride=[self.stride] * 3, padding=self.padding)
        return out


class pair_net(nn.Module):
    def __init__(self, encoder_dim=64):
        super(pair_net, self).__init__()
        self.encoder_dim = encoder_dim
        kernel_width = 3
        self.conv1 = nn.Conv3d(1, self.encoder_dim // 4, padding=1,
                               kernel_size=(kernel_width, kernel_width, kernel_width))
        self.ln1 = nn.BatchNorm3d(self.encoder_dim // 4)
        self.conv2 = nn.Conv3d(self.encoder_dim // 4, self.encoder_dim // 2, padding=1,
                               kernel_size=(kernel_width, kernel_width, kernel_width))
        self.ln2 = nn.BatchNorm3d(self.encoder_dim // 2)

        self.fc = nn.Conv1d(self.encoder_dim // 2, self.encoder_dim, 1)
        self.ln = nn.BatchNorm1d(self.encoder_dim)

        self.conv3 = nn.Conv3d(self.encoder_dim // 2, self.encoder_dim, padding=1,
                               kernel_size=(kernel_width, kernel_width, kernel_width))
        self.ln3 = nn.BatchNorm3d(self.encoder_dim)
        self.conv4 = nn.Conv3d(self.encoder_dim, self.encoder_dim, padding=1,
                               kernel_size=(kernel_width, kernel_width, kernel_width))
        self.ln4 = nn.BatchNorm3d(self.encoder_dim)

        self.conv5 = nn.Conv3d(self.encoder_dim, self.encoder_dim * 2,
                               kernel_size=(kernel_width, kernel_width, kernel_width), padding=0)
        self.ln5 = nn.BatchNorm3d(self.encoder_dim * 2)
        self.conv6 = nn.Conv3d(self.encoder_dim * 2, self.encoder_dim * 4,
                               kernel_size=(2, 2, 2), padding=0)
        self.ln6 = nn.BatchNorm3d(self.encoder_dim * 4)

        self.cls = nn.Sequential(nn.Conv1d(self.encoder_dim * 8, self.encoder_dim // 2, 1),
                                 nn.BatchNorm1d(self.encoder_dim // 2),
                                 nn.ReLU(),
                                 nn.Conv1d(self.encoder_dim // 2, 2, 1))
        self.loss_func = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, data):
        data = data[:, 0, :, :, :].unsqueeze(1)
        # B, C, D, _, _ = data.shape
        # data = data.permute(0,2,3,4,1).reshape(-1)
        # data = get_1d_sincos_pos_embed_from_grid(20, data)
        # data = data.reshape(B, D, D, D, -1).permute(0,4,1,2,3)
        f1 = F.relu(self.ln1(self.conv1(data)))
        f2 = F.relu(self.ln2(self.conv2(f1)))

        f3 = F.relu(self.ln3(self.conv3(f2)))
        f4 = F.relu(self.ln4(self.conv4(f3)))

        shape = f2.shape

        f4 = f4 + self.fc(f2.reshape(shape[0], shape[1], -1)).reshape(shape[0], -1, shape[2], shape[3], shape[4])

        f5 = F.relu(self.ln5(self.conv5(f4)))
        f6 = F.relu(self.ln6(self.conv6(f5)))

        feature = f6

        B, C, _, _, _ = feature.shape
        feature = feature.reshape(B, C, -1)
        pair_feature = torch.cat([feature.unsqueeze(2).repeat(1, 1, 8, 1), feature.unsqueeze(3).repeat(1, 1, 1, 8)], 1)
        pair_feature = pair_feature.reshape(B, 2 * C, -1)
        out = self.cls(pair_feature)
        return out

    def cal_loss(self, out, label):
        label = label.long()
        label = torch.clamp(label + 1, max=1)
        pair_label = (label.unsqueeze(1).repeat(1, 8, 1) + label.unsqueeze(2).repeat(1, 1, 8)) % 2

        B, N = out.shape[0], out.shape[2]
        out = out.permute(0, 2, 1).reshape(-1, 2)
        logit = F.log_softmax(out, dim=-1)
        pair_label = pair_label.long().reshape(-1)
        loss = F.nll_loss(logit, pair_label, reduction='none').reshape(B, N).mean()

        acc_head = out.argmax(dim=1).reshape(B, N)
        acc_head = (acc_head == pair_label.reshape(B, N)).float().mean(-1)
        acc = (acc_head + 1e-3).floor().mean()
        return loss, acc

    def predict(self, out, label):
        B, _, N = out.shape
        out = out.permute(0, 2, 1).reshape(-1, 2)
        logit = F.log_softmax(out, dim=-1)
        label = label.reshape(-1).long()
        label = torch.clamp(label + 1, max=1)
        confidence = (F.softmax(out, -1)[:, 0] - 0.5).abs()
        confidence = confidence.reshape(B, N).min(-1)[0]
        pred = logit.argmax(dim=1).reshape(B, N)
        sign = (pred - 0.5) * 2
        sign[torch.where(confidence < 0.1)] = 1

        acc_head = logit.argmax(dim=1).reshape(B, N)
        acc_tail = logit.argmin(dim=1).reshape(B, N)
        label = label.reshape(B, N)
        acc_head = (acc_head == label).float().mean(-1)
        acc_tail = (acc_tail == label).float().mean(-1)
        acc = torch.stack([acc_head, acc_tail]).max(0)[0].mean()
        return sign, acc


class vertex_net2(nn.Module):
    def __init__(self, encoder_dim=64, sepa=True):
        super(vertex_net2, self).__init__()
        self.encoder_dim = encoder_dim
        kernel_width = 3
        self.conv1 = nn.Conv3d(4, self.encoder_dim // 4, padding=0,
                               kernel_size=(kernel_width, kernel_width, kernel_width))
        self.ln1 = nn.BatchNorm3d(self.encoder_dim // 4)
        self.conv2 = nn.Conv3d(self.encoder_dim // 4, self.encoder_dim // 2, padding=1,
                               kernel_size=(kernel_width, kernel_width, kernel_width))
        self.ln2 = nn.BatchNorm3d(self.encoder_dim // 2)

        self.fc = nn.Conv1d(self.encoder_dim // 2, self.encoder_dim, 1)
        self.ln = nn.BatchNorm1d(self.encoder_dim)

        self.conv3 = nn.Conv3d(self.encoder_dim // 2, self.encoder_dim, padding=1,
                               kernel_size=(kernel_width, kernel_width, kernel_width))
        self.ln3 = nn.BatchNorm3d(self.encoder_dim)
        self.conv4 = nn.Conv3d(self.encoder_dim, self.encoder_dim, padding=0,
                               kernel_size=(kernel_width, kernel_width, kernel_width))
        self.ln4 = nn.BatchNorm3d(self.encoder_dim)

        self.conv5 = nn.Conv3d(self.encoder_dim, self.encoder_dim,
                               kernel_size=(kernel_width, kernel_width, kernel_width), padding=1)
        self.ln5 = nn.BatchNorm3d(self.encoder_dim)
        self.conv6 = nn.Conv3d(self.encoder_dim, self.encoder_dim * 2,
                               kernel_size=(kernel_width, kernel_width, kernel_width), padding=0)
        self.ln6 = nn.BatchNorm3d(self.encoder_dim * 2)

        # self.conv7 = nn.Conv3d(self.encoder_dim, self.encoder_dim,
        #                        kernel_size=(kernel_width, kernel_width, kernel_width))
        # self.ln7 = nn.BatchNorm3d(self.encoder_dim)
        # self.conv8 = nn.Conv3d(self.encoder_dim, self.encoder_dim,
        #                        kernel_size=(kernel_width, kernel_width, kernel_width))
        # self.ln8 = nn.BatchNorm3d(self.encoder_dim)
        self.sepa = sepa
        # self.drop = nn.Dropout(0.4)

        self.cls = nn.Sequential(nn.Conv1d(self.encoder_dim  + 60, self.encoder_dim // 2, 1),
                                 nn.BatchNorm1d(self.encoder_dim // 2),
                                 nn.ReLU(),
                                 nn.Conv1d(self.encoder_dim // 2, 2, 1))

    def forward(self, data):
        # data = data[:, 0, :, :, :].unsqueeze(1)
        # data[:, 0, :, :, :] = data[:, 0, :, :, :]/128
        # B, C, D, _, _ = data.shape
        # data = data.permute(0,2,3,4,1).reshape(-1)
        # data = get_1d_sincos_pos_embed_from_grid(20, data)
        # data = data.reshape(B, D, D, D, -1).permute(0,4,1,2,3)
        f1 = F.relu(self.ln1(self.conv1(data)))
        f2 = F.relu(self.ln2(self.conv2(f1)))

        f3 = F.relu(self.ln3(self.conv3(f2)))
        f4 = F.relu(self.ln4(self.conv4(f3)))

        # shape = f2.shape
        #
        # f4 = f4 + self.fc(f2.reshape(shape[0], shape[1], -1)).reshape(shape[0], -1, shape[2], shape[3], shape[4])[:,:,1:4,1:4,1:4]
        #
        # f5 = F.relu(self.ln5(self.conv5(f4)))
        # f6 = F.relu(self.ln6(self.conv6(f5)))
        #
        # f7 = F.relu(self.ln7(self.conv7(F.pad(f6, (1, 1, 1, 1, 1, 1), 'constant', 0.0))))
        # f8 = F.relu(self.ln8(self.conv8(F.pad(f7, (1, 1, 1, 1, 1, 1), 'constant', 0.0))))

        feature = f4
        B, C, _, _, _ = feature.shape
        xyz = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]],
                           dtype=torch.float32, device=feature.device)
        xyz = xyz.unsqueeze(0).repeat(B, 1, 1)
        xyz_embed = get_3d_sincos_pos_embed_from_point(60, xyz)
        feature = feature.squeeze().unsqueeze(1).repeat(1, 8, 1)
        feature = torch.cat([feature, xyz_embed], -1).permute(0, 2, 1)
        out = self.cls(feature)
        return out

    def cal_loss(self, out, label):

        label = label.long()
        label = torch.clamp(label + 1, max=1)
        if self.sepa:
            B, N = out.shape[0], out.shape[2]
            out = out.permute(0, 2, 1).reshape(-1, 2)
            logit = F.log_softmax(out, dim=-1)
            label = label.long().reshape(-1)
            loss = F.nll_loss(logit, label, reduction='none').reshape(B, N).mean(-1)
            loss_sys = F.nll_loss(logit, (label + 1) % 2, reduction='none').reshape(B, N).mean(-1)

            acc_head = out.argmax(dim=1).reshape(B, N)
            acc_tail = out.argmin(dim=1).reshape(B, N)
        else:
            loss = self.loss_func(out, label.float()).mean(-1)
            loss_sys = self.loss_func(out, ((label + 1) % 2).float()).mean(-1)

            acc_head = (out > 0.0).int()
            acc_tail = (out < 0.0).int()

        acc_head = (acc_head == label.reshape(B, N)).float().mean(-1)
        acc_tail = (acc_tail == label.reshape(B, N)).float().mean(-1)
        acc = (torch.stack([acc_head, acc_tail]).max(0)[0] + 1e-3).floor().mean()

        # loss_com = torch.stack([loss_sys, loss]).min(0)[0].mean()
        loss_com = loss.mean()
        return loss_com, acc

    def predict(self, out, label):
        B, _, N = out.shape
        out = out.permute(0, 2, 1).reshape(-1, 2)
        logit = F.log_softmax(out, dim=-1)
        label = label.reshape(-1).long()
        label = torch.clamp(label + 1, max=1)
        confidence = (F.softmax(out, -1)[:, 0] - 0.5).abs()
        confidence = confidence.reshape(B, N).min(-1)[0]
        pred = logit.argmax(dim=1).reshape(B, N)
        sign = (pred - 0.5) * 2
        # sign[torch.where(confidence<0.3)] = 1

        acc_head = logit.argmax(dim=1).reshape(B, N)
        acc_tail = logit.argmin(dim=1).reshape(B, N)
        label = label.reshape(B, N)
        acc_head = (acc_head == label).float().mean(-1)
        acc_tail = (acc_tail == label).float().mean(-1)
        acc = (torch.stack([acc_head, acc_tail]).max(0)[0] + 1e-3).floor().mean()
        return sign, acc


class vertex_net(nn.Module):
    def __init__(self, encoder_dim=64, sepa=True):
        super(vertex_net, self).__init__()
        self.encoder_dim = encoder_dim
        kernel_width = 3
        self.conv1 = nn.Conv3d(1, self.encoder_dim // 4, kernel_size=(kernel_width, kernel_width, kernel_width), padding=0)
        self.ln1 = nn.BatchNorm3d(self.encoder_dim // 4)
        self.conv2 = nn.Conv3d(self.encoder_dim // 4, self.encoder_dim//2,
                               kernel_size=(kernel_width, kernel_width, kernel_width), padding=1)
        self.ln2 = nn.BatchNorm3d(self.encoder_dim//2)

        self.fc = nn.Conv1d(self.encoder_dim//2, self.encoder_dim, 1)
        self.ln = nn.BatchNorm1d(self.encoder_dim)

        self.conv3 = nn.Conv3d(self.encoder_dim//2, self.encoder_dim,
                               kernel_size=(kernel_width, kernel_width, kernel_width), padding=1)
        self.ln3 = nn.BatchNorm3d(self.encoder_dim)
        self.conv4 = nn.Conv3d(self.encoder_dim, self.encoder_dim,
                               kernel_size=(kernel_width, kernel_width, kernel_width), padding=1)
        self.ln4 = nn.BatchNorm3d(self.encoder_dim)

        self.conv5 = nn.Conv3d(self.encoder_dim, self.encoder_dim,
                               kernel_size=(kernel_width, kernel_width, kernel_width), padding=1)
        self.ln5 = nn.BatchNorm3d(self.encoder_dim)
        self.conv6 = nn.Conv3d(self.encoder_dim, self.encoder_dim,
                               kernel_size=(kernel_width, kernel_width, kernel_width), padding=1)
        self.ln6 = nn.BatchNorm3d(self.encoder_dim)

        self.conv7 = nn.Conv3d(self.encoder_dim, self.encoder_dim,
                               kernel_size=(kernel_width, kernel_width, kernel_width), padding=1)
        self.ln7 = nn.BatchNorm3d(self.encoder_dim)
        self.conv8 = nn.Conv3d(self.encoder_dim, self.encoder_dim,
                               kernel_size=(kernel_width, kernel_width, kernel_width), padding=1)
        self.ln8 = nn.BatchNorm3d(self.encoder_dim)

        self.cls = nn.Sequential(nn.Conv1d(self.encoder_dim * 2, self.encoder_dim // 2, 1),
                                 nn.BatchNorm1d(self.encoder_dim // 2),
                                 nn.ReLU(),
                                 nn.Conv1d(self.encoder_dim // 2, 2, 1))

    def forward(self, data):
        # B, C, D, _, _ = data.shape
        # data = data.permute(0,2,3,4,1).reshape(-1)
        # data = get_1d_sincos_pos_embed_from_grid(20, data)
        # data = data.reshape(B, D, D, D, -1).permute(0,4,1,2,3)
        f1 = F.relu(self.ln1(self.conv1(data)))
        f2 = F.relu(self.ln2(self.conv2(f1)))

        f3 = F.relu(self.ln3(self.conv3(f2)))
        f4 = F.relu(self.ln4(self.conv4(f3)))

        shape = f2.shape

        f4 = f4 + self.fc(f2.reshape(shape[0], shape[1], -1)).reshape(shape[0], -1, shape[2], shape[3], shape[4])

        f5 = F.relu(self.ln5(self.conv5(f4)))
        f6 = F.relu(self.ln6(self.conv6(f5)))
        f6 = f6 + f4
        f7 = F.relu(self.ln7(self.conv7(f6)))
        f8 = F.relu(self.ln8(self.conv8(f7)))
        #
        feature_conv = f8
        # feature_conv = f4
        B, C, _, _, _ = feature_conv.shape
        data_width = feature_conv.shape[-1]
        feature = feature_conv[:, :, data_width // 2 - 1:data_width // 2 + 2:2,
                  data_width // 2 - 1:data_width // 2 + 2:2, data_width // 2 - 1:data_width // 2 + 2:2]
        feature_rel = torch.cat(
            [feature, feature[:, :, 0, 0, 0].unsqueeze(2).unsqueeze(2).unsqueeze(2).repeat(1, 1, 2, 2, 2)], 1)
        feature = feature_rel.reshape(B, 2 * C, 8)  # .permute(0,2,1)

        out = self.cls(feature)
        return out

    def cal_loss(self, out, label):
        label = label.long()
        label = torch.clamp(label + 1, max=1)

        B, N = out.shape[0], out.shape[2]
        out = out.permute(0, 2, 1).reshape(-1, 2)
        logit = F.log_softmax(out, dim=-1)
        label = label.long().reshape(-1)
        # label = ((label[:,1:] + label[:, 0].unsqueeze(1))%2).reshape(-1)
        loss = F.nll_loss(logit, label, reduction='none').reshape(B, N).mean(-1)
        loss_sys = F.nll_loss(logit, (label + 1) % 2, reduction='none').reshape(B, N).mean(-1)

        # pred_label = out.argmax(dim=1).reshape(B, N)
        # label = label.reshape(B, N)
        # acc = (label == pred_label).float().mean()
        acc_head = out.argmax(dim=1).reshape(B, N)
        acc_tail = out.argmin(dim=1).reshape(B, N)

        acc_head = (acc_head == label.reshape(B, N)).float().mean(-1)
        acc_tail = (acc_tail == label.reshape(B, N)).float().mean(-1)
        acc = (torch.stack([acc_head, acc_tail]).max(0)[0] + 1e-3).floor().mean()

        # loss_com = torch.stack([loss_sys, loss]).min(0)[0].mean()
        loss_com = loss.mean()
        return loss_com, acc

    def predict(self, out, label):
        B, _, N = out.shape
        out = out.permute(0, 2, 1).reshape(-1, 2)
        logit = F.log_softmax(out, dim=-1)
        label = label.reshape(-1).long()
        label = torch.clamp(label + 1, max=1)
        confidence = (F.softmax(out, -1)[:, 0] - 0.5).abs()
        confidence = confidence.reshape(B, N).min(-1)[0]
        pred = logit.argmax(dim=1).reshape(B, N)
        sign = (pred - 0.5) * 2
        sign[torch.where(confidence < 0.1)] = 1

        acc_head = logit.argmax(dim=1).reshape(B, N)
        acc_tail = logit.argmin(dim=1).reshape(B, N)
        label = label.reshape(B, N)
        acc_head = (acc_head == label).float().mean(-1)
        acc_tail = (acc_tail == label).float().mean(-1)
        acc = (torch.stack([acc_head, acc_tail]).max(0)[0] + 1e-3).floor().mean()
        return sign, acc


class dual_vertex_net(nn.Module):
    def __init__(self, encoder_dim=64, sepa=True):
        super(dual_vertex_net, self).__init__()
        self.encoder_dim = encoder_dim
        kernel_width = 3
        self.conv1 = dual_conv(1, self.encoder_dim // 4, kernel_size=(kernel_width, kernel_width, kernel_width))
        self.ln1 = nn.BatchNorm3d(self.encoder_dim // 4)
        self.conv2 = dual_conv(self.encoder_dim // 4, self.encoder_dim // 2,
                               kernel_size=(kernel_width, kernel_width, kernel_width))
        self.ln2 = nn.BatchNorm3d(self.encoder_dim // 2)

        self.fc = nn.Conv1d(self.encoder_dim // 2, self.encoder_dim, 1)
        self.ln = nn.BatchNorm1d(self.encoder_dim)

        self.conv3 = dual_conv(self.encoder_dim // 2, self.encoder_dim,
                               kernel_size=(kernel_width, kernel_width, kernel_width))
        self.ln3 = nn.BatchNorm3d(self.encoder_dim)
        self.conv4 = dual_conv(self.encoder_dim, self.encoder_dim,
                               kernel_size=(kernel_width, kernel_width, kernel_width))
        self.ln4 = nn.BatchNorm3d(self.encoder_dim)

        self.conv5 = dual_conv(self.encoder_dim, self.encoder_dim,
                               kernel_size=(kernel_width, kernel_width, kernel_width), padding=0)
        self.ln5 = nn.BatchNorm3d(self.encoder_dim)
        self.conv6 = dual_conv(self.encoder_dim, self.encoder_dim,
                               kernel_size=(kernel_width, kernel_width, kernel_width), padding=1)
        self.ln6 = nn.BatchNorm3d(self.encoder_dim)

        self.cls = nn.Sequential(nn.Conv1d(self.encoder_dim * 2, self.encoder_dim // 2, 1),
                                 nn.BatchNorm1d(self.encoder_dim // 2),
                                 nn.ReLU(),
                                 nn.Conv1d(self.encoder_dim // 2, 2, 1))

    def forward(self, data):
        data = data[:, 0, :, :, :].unsqueeze(1)
        # B, C, D, _, _ = data.shape
        # data = data.permute(0,2,3,4,1).reshape(-1)
        # data = get_1d_sincos_pos_embed_from_grid(20, data)
        # data = data.reshape(B, D, D, D, -1).permute(0,4,1,2,3)
        f1 = F.relu(self.ln1(self.conv1(data)))
        f2 = F.relu(self.ln2(self.conv2(f1)))

        f3 = F.relu(self.ln3(self.conv3(f2)))
        f4 = F.relu(self.ln4(self.conv4(f3)))

        shape = f2.shape

        f4 = f4 + self.fc(f2.reshape(shape[0], shape[1], -1)).reshape(shape[0], -1, shape[2], shape[3], shape[4])

        f5 = F.relu(self.ln5(self.conv5(f4)))
        f6 = F.relu(self.ln6(self.conv6(f5)))

        feature_conv = f6
        B, C, _, _, _ = feature_conv.shape
        data_width = feature_conv.shape[-1]
        feature = feature_conv[:, :, data_width // 2 - 1:data_width // 2 + 2:2,
                  data_width // 2 - 1:data_width // 2 + 2:2, data_width // 2 - 1:data_width // 2 + 2:2]
        feature = feature.reshape(B, C, 8)  # .permute(0,2,1)
        anchor_feature = feature_conv[:, :, 1, 1, 1].unsqueeze(2).repeat(1, 1, 8)
        feature = torch.cat([feature, anchor_feature], 1)
        # feature_ref = feature[:,:,0].unsqueeze(2).repeat(1,1,7)
        # # feature = torch.cat([feature[:,:,1:], feature_ref], 1)
        # feature = feature[:,:,1:] - feature_ref
        # feature = self.drop(feature)
        out = self.cls(feature)
        return out

    def cal_loss(self, out, label):
        label = label.long()
        label = torch.clamp(label + 1, max=1)

        B, N = out.shape[0], out.shape[2]
        out = out.permute(0, 2, 1).reshape(-1, 2)
        logit = F.log_softmax(out, dim=-1)
        label = label.long().reshape(-1)
        # label = ((label[:,1:] + label[:, 0].unsqueeze(1))%2).reshape(-1)
        loss = F.nll_loss(logit, label, reduction='none').reshape(B, N).mean(-1)
        loss_sys = F.nll_loss(logit, (label + 1) % 2, reduction='none').reshape(B, N).mean(-1)

        # pred_label = out.argmax(dim=1).reshape(B, N)
        # label = label.reshape(B, N)
        # acc = (label == pred_label).float().mean()
        acc_head = out.argmax(dim=1).reshape(B, N)
        acc_tail = out.argmin(dim=1).reshape(B, N)

        acc_head = (acc_head == label.reshape(B, N)).float().mean(-1)
        acc_tail = (acc_tail == label.reshape(B, N)).float().mean(-1)
        acc = (torch.stack([acc_head, acc_tail]).max(0)[0] + 1e-3).floor().mean()

        loss_com = torch.stack([loss_sys, loss]).min(0)[0].mean()
        return loss_com, acc

    def predict(self, out, label):
        B, _, N = out.shape
        out = out.permute(0, 2, 1).reshape(-1, 2)
        logit = F.log_softmax(out, dim=-1)
        label = label.reshape(-1).long()
        label = torch.clamp(label + 1, max=1)
        confidence = (F.softmax(out, -1)[:, 0] - 0.5).abs()
        confidence = confidence.reshape(B, N).min(-1)[0]
        pred = logit.argmax(dim=1).reshape(B, N)
        sign = (pred - 0.5) * 2
        sign[torch.where(confidence < 0.1)] = 1

        acc_head = logit.argmax(dim=1).reshape(B, N)
        acc_tail = logit.argmin(dim=1).reshape(B, N)
        label = label.reshape(B, N)
        acc_head = (acc_head == label).float().mean(-1)
        acc_tail = (acc_tail == label).float().mean(-1)
        acc = (torch.stack([acc_head, acc_tail]).max(0)[0] + 1e-3).floor().mean()
        return sign, acc


if __name__ == '__main__':
    data = torch.Tensor(32, 3, 5, 5, 5)
    d_conv = dual_conv(3, 8, 3, 1, 1)
    conv = nn.Conv3d(3, 8, kernel_size=(3, 3, 3), padding=1, stride=1)
    out = d_conv(data)
    s = 0
