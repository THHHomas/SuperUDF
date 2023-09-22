from utils import index_points, square_distance, knn_point
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from transformer_pos_enc import get_1d_sincos_pos_embed_from_grid
from transformer_pos_enc import get_3d_sincos_pos_embed_from_point

class TLinear(nn.Module):
    def __init__(self, d_in, d_out, bias=True) -> None:
        super().__init__()
        self.linear_c = nn.Linear(d_in, d_out*d_in)
        self.W = nn.Parameter(torch.Tensor(d_in))
        self.mask = nn.Parameter(torch.Tensor(d_in, d_in))
        nn.init.kaiming_uniform_(self.mask, a=math.sqrt(5))
        # self.bn = nn.BatchNorm1d(d_in)
        nn.init.constant_(self.W, 1)
        self.bias = bias
        if bias:
            self.b = nn.Parameter(torch.Tensor(d_out))
            nn.init.constant_(self.b, 0)
        # self.channel_embedding = nn.Parameter(
        #     torch.from_numpy(get_1d_sincos_pos_embed_from_grid(d_model, np.arange(d_model))).float(), requires_grad=False)
        self.d_out = d_out
        self.d_in = d_in
        # self.pos_embedding = nn.Parameter(torch.Tensor(d_model))
        # nn.init.constant_(self.pos_embedding, 0.0)
        # self.diag = nn.Parameter(torch.diag(torch.ones(d_model)), requires_grad=False)
        # self.sigma = nn.Parameter(torch.tensor(0.0))
    # xyz: b x n x 3, features: b x n x f

    def forward(self, features):
        B, N, C = features.shape
        # features = self.bn(features.permute(0,2,1)).permute(0,2,1)
        # W = self.W/(torch.norm(self.W)+1e-5)*math.sqrt(self.d_in)
        # features *= W
        # relative_feature = features.unsqueeze(2) - features.unsqueeze(3)
        # relative_feature = relative_feature*self.mask
        # # make it sym
        # relative_feature = relative_feature * self.mask
        # relative_feature = relative_feature + relative_feature.permute(0, 1, 3, 2)
        #
        # relative_feature = torch.cat([relative_feature, features.unsqueeze(3).repeat(1, 1, 1, self.d_in)], -1)
        weight_c = self.linear_c(features).reshape(B, N, self.d_out, -1)
        weight_c = weight_c / (weight_c.abs() + 1e-5).sum(-2, keepdim=True)  #*math.sqrt(self.d_in)
        features = torch.matmul(weight_c, features.unsqueeze(2)).squeeze() # /math.sqrt(k.size(-1))
        return features


class MTLinear(nn.Module):
    def __init__(self, d_in, d_out, bias=True) -> None:
        super().__init__()
        self.linear_c = nn.Linear(d_in, d_out*d_in)
        self.W = nn.Parameter(torch.Tensor(d_in))
        self.mask = nn.Parameter(torch.Tensor(d_in, d_in))
        nn.init.kaiming_uniform_(self.mask, a=math.sqrt(5))
        # self.bn = nn.BatchNorm1d(d_in)
        nn.init.constant_(self.W, 1)
        self.bias = bias
        if bias:
            self.b = nn.Parameter(torch.Tensor(d_out))
            nn.init.constant_(self.b, 0)
        self.d_out = d_out
        self.d_in = d_in

    def forward(self, features):
        B, N, C = features.shape
        mfeature = features.mean(1)
        weight_c = self.linear_c(mfeature).reshape(B, self.d_out, -1)
        weight_c = weight_c / (weight_c.abs() + 1e-5).sum(-2, keepdim=True)  #*math.sqrt(self.d_in)
        features = torch.bmm(features, weight_c.permute(0, 2, 1))
        return features


class MLinear(nn.Module):
    def __init__(self, d_in, d_out, bias=True) -> None:
        super().__init__()
        self.linear_c = nn.Linear(d_in, d_out)
        self.W1 = nn.Parameter(torch.Tensor(d_in, d_in))
        self.W2 = nn.Parameter(torch.Tensor(d_in, d_in))
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))
        self.d_out = d_out
        self.d_in = d_in

    def forward(self, features):
        data_shape = features.size()
        data_size = len(data_shape)
        if data_size > 3:
            B, N, K, C = data_shape
            features = features.reshape(B, -1, C)
        if data_size == 3:
            B, N, C = data_shape
        mfeature = features.mean(dim=1)
        weight_c = mfeature.unsqueeze(1)*self.W1 - mfeature.unsqueeze(2)*self.W2
        weight_c = self.linear_c(weight_c) #.reshape(B, self.d_out, -1)
        weight_c = weight_c / (weight_c.abs() + 1e-5).sum(-1, keepdim=True)  #*math.sqrt(self.d_in)
        # features = torch.matmul(weight_c.unsqueeze(1), features.unsqueeze(3)).squeeze() # /math.sqrt(k.size(-1))
        features = torch.bmm(features, weight_c)
        if data_size > 3:
            features = features.reshape(B, N, K, -1)
        return features


class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k=24) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.position_embed_dim = 60
        self.fc_delta = nn.Sequential(
            nn.Linear(self.position_embed_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

    # xyz: b x n x 3, features: b x n x f
    def forward(self, features, xyz, knn_idx, knn_xyz):
        x = self.fc1(features)
        # q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)
        q, k, v = x, index_points(x, knn_idx), index_points(x, knn_idx)
        g_xyz = xyz[:, :, None] - knn_xyz
        g_xyz_embed = get_3d_sincos_pos_embed_from_point(self.position_embed_dim, g_xyz)
        pos_enc = self.fc_delta(g_xyz_embed)  # b x n x k x f

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc )
        res = self.fc2(res) + x
        return res, attn




class TransformerBlock_fc(nn.Module):
    def __init__(self, d_points, d_model, k=24) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.position_embed_dim = 60
        self.fc_delta = nn.Sequential(
            nn.Linear(self.position_embed_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_atten = nn.Linear(d_model*k, d_model*k)
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

    # xyz: b x n x 3, features: b x n x f
    def forward(self, features, xyz):
        B, N, _ = xyz.shape
        with torch.no_grad():
            dists = square_distance(xyz, xyz)
            knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
            knn_xyz = index_points(xyz, knn_idx)

        x = self.fc1(features)
        # q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)
        q, k, v = x, index_points(x, knn_idx), index_points(x, knn_idx)
        g_xyz = xyz[:, :, None] - knn_xyz
        g_xyz_embed = get_3d_sincos_pos_embed_from_point(self.position_embed_dim, g_xyz)
        pos_enc = self.fc_delta(g_xyz_embed)  # b x n x k x f

        # attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = self.fc_atten((q[:, :, None] - k + pos_enc).reshape(B, N, -1)).reshape(B, N, self.k, -1)

        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc )
        res = self.fc2(res) + x
        return res, attn

class CTransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k=24) -> None:
        super().__init__()
        # MLinear = nn.Linear
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 =MLinear(d_model, d_model)
        self.fc3 = MLinear(d_model, d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = MLinear(d_model, d_model)#, bias=False)
        self.w_ks = MLinear(d_model, d_model)#, bias=False)
        self.w_vs = MLinear(d_model, d_model)#, bias=False)
        self.k = k


    def forward(self, features, xyz):
        B, N, C = features.shape
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)

        x = self.fc1(features)
        q, k, v = self.w_qs(x), self.w_ks(x), self.w_vs(x)
        k = index_points(k, knn_idx)
        v = index_points(v, knn_idx)
        # q, k, v = x, index_points(x, knn_idx), index_points(x, knn_idx)
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x f

        # attn = self.fc_gamma(q[:, :, None] - k + pos_enc)

        # attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        # attn = attn / (attn.abs() + 1e-5).sum(-2, keepdim=True)
        attn = attn / (attn.abs() + 1e-5).sum(-1, keepdim=True)
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + x
        # res = self.fc3(res) + res
        return res, attn


class weight_generator(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(weight_generator, self).__init__()
        self.prefix_linear = nn.Linear(input_dim, out_dim)
        self.position_embed_dim = 60
        self.fc_position = nn.Sequential(
            nn.Linear(self.position_embed_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.fc_shift = nn.Sequential(
            nn.Linear(self.position_embed_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        # self.fc_position2 = nn.Sequential(
        #     nn.Linear(self.position_embed_dim, input_dim),
        #     nn.ReLU(),
        #     nn.Linear(input_dim, input_dim)
        # )
        # self.fc_position2 = nn.Linear(out_dim, out_dim)
        # self.fc_matrix = nn.Linear(input_dim, 12)
        # self.linear_merge = nn.Linear(out_dim * 2, out_dim)
        self.linear_t = nn.Sequential(nn.Linear((out_dim*1)+1, out_dim//4),
                                      nn.ReLU(),
                                      nn.Linear(out_dim//4, 1))
        self.linear_g = nn.Linear(2, 1)
        self.linear_w = nn.Linear(out_dim*3, 1)
        # self.linear_t = nn.Sequential(nn.Linear(out_dim+out_dim, out_dim//2), nn.ReLU(), nn.Linear(out_dim//2, out_dim//4),
        #                               nn.ReLU(), nn.Linear(out_dim // 4, out_dim // 8),
        #                               nn.ReLU(), nn.Linear(out_dim // 8, 3))

        self.linear_k = nn.Linear(8, 8)
        self.out_dim = out_dim
        self.fc = nn.Linear(out_dim * 2, out_dim)
        self.acti = nn.Tanh()

    def forward(self, feature1, xyz1, query_xyz, error, point_index, normal=None, fps_num=None, knn_num=12):
        B, N, _ = xyz1.shape
        M = query_xyz.shape[1]
        # query_xyz = torch.stack(query_xyz)

        feature_1 = self.prefix_linear(feature1)

        # point_index = knn_point(knn_num, xyz1, query_xyz, sorted=False)
        error = error.reshape(B, N, 1)
        error = (F.tanh(error)*0.001+0.002)
        g_feature = index_points(feature_1, point_index)
        g_error = index_points(error, point_index)[:,:,:,0]
        # sigma, u = torch.std_mean(g_error)
        #
        g_xyz = index_points(xyz1, point_index)
        g_xyz = g_xyz - query_xyz.unsqueeze(2)

        # g_xyz_embed = get_3d_sincos_pos_embed_from_point(self.position_embed_dim, g_xyz)
        # g_xyz_feature = self.fc_position(g_xyz_embed)
        #
        # shift_chose_embed = get_3d_sincos_pos_embed_from_point(self.position_embed_dim, shift_chose)
        # shift_chose_feature = self.fc_position(shift_chose_embed)
        # distance = torch.norm(g_xyz, dim=-1, keepdim=True)
        # g_cat = torch.cat([g_feature, distance], -1)
        # weight = self.linear_t(g_cat).reshape(B, M, -1)
        # weight = F.softmax(weight, 2)

        distance = torch.norm(g_xyz, dim=-1)
        weight = torch.softmax(-distance*200, -1)
        weight = weight*0.5
        ones = torch.ones_like(weight) * 0.5
        ones[:, :, 1:] = 0
        weight = weight + ones
        return weight



class belong_weight_generator(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(belong_weight_generator, self).__init__()
        self.prefix_linear = nn.Linear(input_dim, out_dim)
        self.position_embed_dim = 60
        self.fc_position = nn.Sequential(
            nn.Linear(self.position_embed_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        self.fc_shift = nn.Sequential(
            nn.Linear(self.position_embed_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        # self.fc_position2 = nn.Sequential(
        #     nn.Linear(self.position_embed_dim, input_dim),
        #     nn.ReLU(),
        #     nn.Linear(input_dim, input_dim)
        # )
        # self.fc_position2 = nn.Linear(out_dim, out_dim)
        # self.fc_matrix = nn.Linear(input_dim, 12)
        # self.linear_merge = nn.Linear(out_dim * 2, out_dim)
        self.linear_t = nn.Sequential(nn.Linear((out_dim*2)*8, out_dim),
                                      nn.ReLU(),
                                      nn.Linear(out_dim, 8))
        self.linear_g = nn.Linear(2, 1)
        self.linear_w = nn.Linear(out_dim*3, 1)
        # self.linear_t = nn.Sequential(nn.Linear(out_dim+out_dim, out_dim//2), nn.ReLU(), nn.Linear(out_dim//2, out_dim//4),
        #                               nn.ReLU(), nn.Linear(out_dim // 4, out_dim // 8),
        #                               nn.ReLU(), nn.Linear(out_dim // 8, 3))

        self.linear_k = nn.Linear(8, 8)
        self.out_dim = out_dim
        self.fc = nn.Linear(out_dim * 2, out_dim)
        self.acti = nn.Tanh()

    def forward(self, feature1, xyz1, query_xyz, error, shift_chose, normal=None, fps_num=None, knn_num=12):
        B, N, _ = xyz1.shape
        M = query_xyz.shape[1]
        # query_xyz = torch.stack(query_xyz)
        feature_1 = self.prefix_linear(feature1)

        point_index = knn_point(knn_num, xyz1, query_xyz)
        # error = error.reshape(B, N, 1)
        # error = (F.tanh(error)*0.02+0.03)
        g_feature = index_points(feature_1, point_index)
        # g_error = index_points(error, point_index)[:,:,:,0]

        g_xyz = index_points(xyz1, point_index)
        g_xyz = g_xyz - query_xyz.unsqueeze(2)

        g_xyz_embed = get_3d_sincos_pos_embed_from_point(self.position_embed_dim, g_xyz)
        g_xyz_feature = self.fc_position(g_xyz_embed)
        g_feature = g_feature + g_xyz_feature
        weight = torch.matmul(g_feature, g_feature[:,:,0].unsqueeze(3))[:,:,:,0]
        weight = torch.softmax(weight, -1)
        weight = weight*0.4
        ones = torch.ones_like(weight)*0.6
        ones[:,:,1:]=0
        weight = weight + ones
        return weight, point_index


class upsample_layer_batch(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(upsample_layer_batch, self).__init__()
        self.prefix_linear = nn.Linear(input_dim, out_dim)
        self.position_embed_dim = 60
        self.fc_position = nn.Sequential(
            nn.Linear(self.position_embed_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        # self.fc_position2 = nn.Linear(out_dim, out_dim)
        # self.fc_matrix = nn.Linear(input_dim, 12)
        # self.linear_merge = nn.Linear(out_dim * 2, out_dim)
        # self.linear_r = nn.Linear(out_dim, 1)
        self.linear_t = nn.Linear(out_dim*2, out_dim)

        self.suffix_linear = nn.Linear(out_dim, out_dim)
        self.out_dim = out_dim
        self.fc = nn.Linear(out_dim * 2, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, feature1, xyz1, query_xyz, normal=None, fps_num=None, knn_num=12):
        B, N, _ = xyz1.shape
        M = query_xyz.shape[1]
        # query_xyz = torch.stack(query_xyz)
        feature_1 = self.prefix_linear(feature1)

        point_index = knn_point(knn_num, xyz1, query_xyz)

        g_feature = index_points(feature_1, point_index)
        # print(query_xyz.shape, point_index.max())
        g_xyz = index_points(xyz1, point_index)


        g_xyz = g_xyz - query_xyz.unsqueeze(2)
        g_xyz_embed = get_3d_sincos_pos_embed_from_point(self.position_embed_dim, g_xyz)
        position_weight = self.fc_position(g_xyz_embed)
        g_feature += position_weight

        position_weight = torch.cat([position_weight, g_feature], -1)
        position_weight = self.linear_t(position_weight)
        # position_weight += add_weight

                # position_weight = position_weight / (position_weight.abs() + 1e-5).sum(-1, keepdim=True)
        position_weight = F.softmax(position_weight, 2)
        # position_weight = position_weight.reshape(B, M, -1)
        # position_weight = position_weight / (position_weight.abs() + 1e-8).sum(-1, keepdim=True)
        # position_weight = position_weight.reshape(B, M, knn_num, -1)*math.sqrt(knn_num)

        new_feature = torch.matmul(position_weight.permute(0, 1, 3, 2).unsqueeze(3),
                                   g_feature.permute(0, 1, 3, 2).unsqueeze(4)).squeeze()
        if B == 1:
            new_feature = new_feature.unsqueeze(0)
        new_feature = self.bn(new_feature.permute(0, 2, 1)).permute(0, 2, 1)
        # g_feature = torch.cat([feature.unsqueeze(1), g_feature], 1)
        # attention_weight = self.linear_t(g_feature-feature.unsqueeze(1))
        # feature = torch.matmul(attention_weight.permute(0,2,1).unsqueeze(2), g_feature.permute(0,2,1).unsqueeze(3)).squeeze()/math.sqrt(knn_num)
        new_feature = new_feature.squeeze()
        return new_feature


class dynamick_indicator_layer_batch(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(dynamick_indicator_layer_batch, self).__init__()
        self.prefix_linear = nn.Linear(input_dim, out_dim)

        self.fc_position = nn.Sequential(
            nn.Linear(3, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        # self.fc_position2 = nn.Linear(out_dim, out_dim)
        self.fc_matrix = nn.Linear(input_dim, 12)
        self.linear_merge = nn.Linear(out_dim * 2, out_dim)
        self.linear_r = nn.Linear(out_dim, 1)
        self.linear_t = nn.Linear(out_dim, out_dim)

        self.suffix_linear = nn.Linear(out_dim, out_dim)
        self.out_dim = out_dim
        self.fc = nn.Linear(out_dim * 2, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, feature1, xyz1, query_xyz, normal=None, fps_num=None, knn_num_min=12, knn_num=12):
        B, N, _ = xyz1.shape
        _, M, _ = query_xyz.shape
        # query_xyz = torch.stack(query_xyz)
        feature_1 = self.prefix_linear(feature1)

        point_index, g_dis = knn_point(knn_num, xyz1, query_xyz, dis=True)
        g_dis = torch.clamp(g_dis[:,:,0], min=0.04, max=0.15)  # closest point
        dynamic_knn_number = ((g_dis-0.04)*(knn_num-knn_num_min)/0.08+knn_num_min).long().reshape(-1)

        g_feature = index_points(feature_1, point_index)

        g_xyz = index_points(xyz1, point_index)
        g_xyz = g_xyz - query_xyz.unsqueeze(2)

        position_weight = self.fc_position(g_xyz)
        g_feature += position_weight
        ###############################dynamic mask generation#################
        mask = torch.zeros(B, M, knn_num, device=g_dis.device, dtype=torch.long).reshape(B*M, -1) #.scatter_(2, dynamic_knn_number-1, 1)
        condition = torch.arange(0, knn_num, device=g_dis.device, dtype=torch.long).unsqueeze(0).repeat(B*M, 1)
        mask[torch.where((condition<dynamic_knn_number.unsqueeze(1)))] = 1
        mask = mask.reshape(B, M, knn_num)
        position_weight = position_weight*mask.unsqueeze(3)
        #######################################################################

        position_weight = position_weight / (position_weight.abs() + 1e-5).sum(-1, keepdim=True)
        new_feature = torch.matmul(position_weight.permute(0, 1, 3, 2).unsqueeze(3),
                                   g_feature.permute(0, 1, 3, 2).unsqueeze(4)).squeeze() / dynamic_knn_number.reshape(B, M, 1).float()
        if B == 1:
            new_feature = new_feature.unsqueeze(0)
        new_feature = self.bn(new_feature.permute(0, 2, 1)).permute(0, 2, 1)
        new_feature = new_feature.squeeze()
        return new_feature



class Grid_TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k=24) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.position_embed_dim = 60
        self.fc_delta = nn.Sequential(
            nn.Linear(self.position_embed_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

    # xyz: b x n x 3, features: b x n x f
    def forward(self, features, xyz):
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)

        x = self.fc1(features)
        # q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)
        q, k, v = x, index_points(x, knn_idx), index_points(x, knn_idx)
        g_xyz = xyz[:, :, None] - knn_xyz
        g_xyz_embed = get_3d_sincos_pos_embed_from_point(self.position_embed_dim, g_xyz)
        pos_enc = self.fc_delta(g_xyz_embed)  # b x n x k x f

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc )
        res = self.fc2(res) + x
        return res, attn