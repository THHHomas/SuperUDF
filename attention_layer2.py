import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from libkdtree import KDTree
except:
    pass
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from utils import index_points, knn_point, farthest_point_sample
from transformer import TransformerBlock, TLinear, upsample_layer_batch, weight_generator
from transformer_pos_enc import get_1d_sincos_pos_embed_from_grid
# from pointnet2_ops import pointnet2_utils
from transformer_pos_enc import get_3d_sincos_pos_embed_from_point
from utils import farthest_point_sample, chamfer_loss, square_distance,\
    chamfer_loss_chunk_no_grad_closest, PointNetFeaturePropagation, chamfer_loss_chunk_no_grad, \
    chamfer_loss_chunk_no_grad_withnrom, chamfer_loss_chunk_no_grad_weighted, interpolation, chamfer_loss_chunk_local

class denoise_encoder(nn.Module):
    def __init__(self, ef_dim, basic_layer, scale):
        super(denoise_encoder, self).__init__()
        scale_1, scale_2 = scale
        self.ef_dim = ef_dim
        self.tl1 = basic_layer(3, self.ef_dim // 4)
        self.bn_tl1 = nn.BatchNorm1d(self.ef_dim // 4)
        self.tl2 = basic_layer(self.ef_dim // 4, self.ef_dim)
        self.bn_tl2 = nn.BatchNorm1d(self.ef_dim)
        self.res_fc = nn.Linear(self.ef_dim, self.ef_dim)
        self.tl3 = basic_layer(self.ef_dim, self.ef_dim)
        self.bn_tl3 = nn.BatchNorm1d(self.ef_dim)
        self.tl3a = basic_layer(self.ef_dim, self.ef_dim)
        self.bn_tl3a = nn.BatchNorm1d(self.ef_dim)
        self.tl3b = basic_layer(self.ef_dim, self.ef_dim)
        self.bn_tl3b = nn.BatchNorm1d(self.ef_dim)

        self.tl4 = basic_layer(self.ef_dim, self.ef_dim)
        self.bn_tl4 = nn.BatchNorm1d(self.ef_dim)

        self.tl5 = basic_layer(self.ef_dim, self.ef_dim*scale_2)
        self.bn_tl5 = nn.BatchNorm1d(self.ef_dim*scale_2)
        self.tl6 = basic_layer(self.ef_dim*scale_2, self.ef_dim*scale_2)
        self.bn_tl6 = nn.BatchNorm1d(self.ef_dim*scale_2)


    def forward(self, xyz, fps_idx1):
        feature1 = self.tl1(xyz, xyz)[0]
        feature1 = self.bn_tl1(feature1.permute(0, 2, 1)).permute(0, 2, 1)
        feature2 = self.tl2(feature1, xyz)[0]
        feature2 = self.bn_tl2(feature2.permute(0, 2, 1)).permute(0, 2, 1)
        feature3 = self.tl3(feature2, xyz)[0]
        feature3 = self.bn_tl3(feature3.permute(0, 2, 1)).permute(0, 2, 1)

        feature4 = self.tl4(feature3, xyz)[0]
        feature4 = self.bn_tl4(feature4.permute(0, 2, 1)).permute(0, 2, 1)
        feature4 += self.res_fc(feature3)

        feature2_1000 = index_points(feature3, fps_idx1)
        xyz2_1000 = index_points(xyz, fps_idx1)

        # *************** 1000 point cloud ***************#
        feature5 = self.tl5(feature2_1000, xyz2_1000)[0]
        feature5 = self.bn_tl5(feature5.permute(0, 2, 1)).permute(0, 2, 1)
        feature6 = self.tl6(feature5, xyz2_1000)[0]
        feature6 = self.bn_tl6(feature6.permute(0, 2, 1)).permute(0, 2, 1)

        return feature4, feature6, fps_idx1


class encoder(nn.Module):
    def __init__(self, ef_dim, basic_layer, scale):
        super(encoder, self).__init__()
        scale_1, scale_2 = scale
        self.ef_dim = ef_dim
        self.tl1 = basic_layer(3, self.ef_dim // 16)
        self.bn_tl1 = nn.BatchNorm1d(self.ef_dim // 16)
        self.tl2 = basic_layer(self.ef_dim // 16, self.ef_dim//8)
        self.bn_tl2 = nn.BatchNorm1d(self.ef_dim//8)
        self.res_fc = nn.Linear(self.ef_dim//8, self.ef_dim*1)
        self.tl3 = basic_layer(self.ef_dim//8, self.ef_dim//4)
        self.bn_tl3 = nn.BatchNorm1d(self.ef_dim//4)
        # self.tl3a = basic_layer(self.ef_dim//2, self.ef_dim)
        # self.bn_tl3a = nn.BatchNorm1d(self.ef_dim)
        # self.tl3b = basic_layer(self.ef_dim, self.ef_dim//2)
        # self.bn_tl3b = nn.BatchNorm1d(self.ef_dim//2)

        self.tl4 = basic_layer(self.ef_dim//4, self.ef_dim*1)
        self.bn_tl4 = nn.BatchNorm1d(self.ef_dim*1)

        # self.tl5 = basic_layer(self.ef_dim, self.ef_dim)
        # self.bn_tl5 = nn.BatchNorm1d(self.ef_dim)
        # self.tl6 = basic_layer(self.ef_dim*scale_2, self.ef_dim*scale_2)
        # self.bn_tl6 = nn.BatchNorm1d(self.ef_dim*scale_2)
        #
        # self.tl7 = basic_layer(self.ef_dim*scale_2, self.ef_dim*scale_1)
        # self.bn_tl7 = nn.BatchNorm1d(self.ef_dim*scale_1)


    def forward(self, xyz):
        # fps_idx = farthest_point_sample(xyz, 500)
        with torch.no_grad():
            dists = square_distance(xyz, xyz)
            knn_idx = dists.argsort()[:, :, :24]  # b x n x k
            knn_xyz = index_points(xyz, knn_idx)

        feature1 = self.tl1(xyz, xyz, knn_idx, knn_xyz)[0]
        feature1 = self.bn_tl1(feature1.permute(0, 2, 1)).permute(0, 2, 1)
        feature2 = self.tl2(feature1, xyz, knn_idx, knn_xyz)[0]
        feature2 = self.bn_tl2(feature2.permute(0, 2, 1)).permute(0, 2, 1)

        feature3 = self.tl3(feature2, xyz, knn_idx, knn_xyz)[0]
        feature3 = self.bn_tl3(feature3.permute(0, 2, 1)).permute(0, 2, 1)

        # xyz_500 = index_points(xyz, fps_idx)
        # feature_500 = index_points(feature3, fps_idx)
        #
        # feature3a = self.tl3a(feature_500, xyz_500)[0]
        # feature3a = self.bn_tl3a(feature3a.permute(0, 2, 1)).permute(0, 2, 1)
        # feature3b = self.tl3b(feature3a, xyz_500)[0]
        # feature3b = self.bn_tl3b(feature3b.permute(0, 2, 1)).permute(0, 2, 1)
        #
        # feature_upsampled = interpolation(xyz_500, xyz, feature3b)

        feature4 = self.tl4(feature3, xyz, knn_idx, knn_xyz)[0]
        feature4 = self.bn_tl4(feature4.permute(0, 2, 1)).permute(0, 2, 1)
        feature4 += self.res_fc(feature2)

        # feature5 = self.tl5(feature4, xyz)[0]
        # feature5 = self.bn_tl5(feature5.permute(0, 2, 1)).permute(0, 2, 1)
        # feature2_1000 = index_points(feature3, fps_idx1)
        # xyz2_1000 = index_points(xyz, fps_idx1)
        #
        # # *************** 1000 point cloud ***************#
        # feature5 = self.tl5(feature2_1000, xyz2_1000)[0]
        # feature5 = self.bn_tl5(feature5.permute(0, 2, 1)).permute(0, 2, 1)
        # feature6 = self.tl6(feature5, xyz2_1000)[0]
        # feature6 = self.bn_tl6(feature6.permute(0, 2, 1)).permute(0, 2, 1)

        return feature4, feature4, feature4


class decoder(nn.Module):
    def __init__(self, input_dim, decoder_dim, basic_layer, scale, knn_list=[12,12,12], dim_scale=[1,1,1]):
        super(decoder, self).__init__()
        scale_1, scale_2 = scale
        self.upsample0 = upsample_layer_batch(input_dim*scale_1, input_dim*scale_2)
        self.upsample1 = upsample_layer_batch(input_dim*scale_2, input_dim)
        # self.indicator200 = upsample_layer_batch(input_dim*scale_1, decoder_dim*dim_scale[0])
        self.indicator1000 = upsample_layer_batch(input_dim*scale_2, decoder_dim*dim_scale[1])
        # self.indicator3000 = upsample_layer_batch(input_dim*(1+dim_scale[1]), decoder_dim*dim_scale[2])
        self.indicator3000 = upsample_layer_batch(input_dim, decoder_dim * dim_scale[2])
        self.tl8 = basic_layer(input_dim*scale_2, input_dim*scale_2)
        self.bn_tl8 = nn.BatchNorm1d(input_dim*scale_2)
        self.tl9 = basic_layer(input_dim, input_dim)
        self.bn_tl9 = nn.BatchNorm1d(input_dim)
        self.knn_list = knn_list

        # self.upsample0 = PointNetFeaturePropagation(input_dim, [input_dim, input_dim])
        # self.upsample1 = PointNetFeaturePropagation(input_dim, [input_dim, input_dim])
        # self.indicator3000 = PointNetFeaturePropagation(input_dim, [decoder_dim, decoder_dim*2])

        self.cls = nn.Sequential(nn.Linear(decoder_dim, decoder_dim // 2),
                                 nn.ReLU(),
                                 nn.Linear(decoder_dim // 2, 3),
                                 nn.Tanh())

    def forward(self, feature4, feature5, feature6, xyz, detect_point):
        # xyz2_1000 = index_points(xyz, fps_idx1)
        # # feature7_up = self.upsample0(xyz2_1000.permute(0, 2, 1), xyz2_200.permute(0, 2, 1), None,
        # #                              feature7.permute(0, 2, 1)).permute(0, 2, 1)
        # feature7_up = feature6
        # feature8 = self.tl8(feature7_up, xyz2_1000)[0]
        # feature8 = self.bn_tl8(feature8.permute(0, 2, 1)).permute(0, 2, 1)
        # # new_feature1000 = self.indicator1000(feature8, xyz2_1000, detect_point, knn_num=self.knn_list[1])
        # feature8_up = self.upsample1(feature8, xyz2_1000, xyz, knn_num=12)
        # # feature8_up = self.upsample1(xyz.permute(0, 2, 1), xyz2_1000.permute(0, 2, 1), None,
        # #                                      feature8.permute(0, 2, 1)).permute(0, 2, 1)
        # if len(feature8_up.shape)==2:
        #     feature8_up = feature8_up.unsqueeze(0)
        # feature8_up = torch.cat([feature8_up, feature4], -1)
        # try:
        new_feature3000 = self.indicator3000(feature5, xyz, detect_point, knn_num=self.knn_list[2])
        # except:
            # print("*************",feature5.shape, xyz.shape, detect_point.shape, self.knn_list[2])
        # new_feature = torch.cat([new_feature3000, new_feature1000], -1)
        # new_feature3000 = self.indicator3000(detect_point.permute(0,2,1), xyz.permute(0,2,1), None, feature9.permute(0,2,1)).permute(0,2,1)
        shift = self.cls(new_feature3000)
        return shift


class recon_net(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, basic_layer, scale):
        super(recon_net, self).__init__()
        # self.encoder = encoder
        self.decoder = decoder(encoder_dim, decoder_dim, basic_layer, scale, knn_list=[0, 12, 24], dim_scale=[1,1,2])
        self.decoder_far = decoder(encoder_dim, decoder_dim, basic_layer, scale, knn_list=[0, 12, 12], dim_scale=[1,1,1])
        self.training = True

    def forward(self, point_cloud, detect_point, fps_idx1, fps_idx2, coding):
        ######################################################
        B, N, _ = detect_point.shape
        dis = ((detect_point.unsqueeze(2) - point_cloud.unsqueeze(1)) ** 2).sum(-1).sqrt().min(-1)[0]
        g_dis, group_idx = torch.topk(dis, N, dim=-1, largest=False, sorted=True)
        ss = g_dis[:,N*2//3].mean()
        close_point = index_points(detect_point, group_idx[:, 0:N*2//3])
        far_point = index_points(detect_point, group_idx[:, N*2//3:])
        logit_close = self.decoder(*(coding+(point_cloud, close_point)))
        logit_far = self.decoder_far(*(coding + (point_cloud, far_point)))
        logit = torch.zeros(B, N, 2, device=point_cloud.device, dtype=torch.float)
        group_idx = group_idx.unsqueeze(2).repeat(1,1,2)
        logit.scatter_(1, group_idx[:, 0:N*2//3], logit_close)
        logit.scatter_(1, group_idx[:, N*2//3:], logit_far)
        ######################################################
        # logit = self.decoder(*(coding + (point_cloud, detect_point)))
        return logit


class bsp_network(nn.Module):
    def __init__(self, phase, scale=[4, 2], voxel_dim=256):
        super(bsp_network, self).__init__()
        self.noise_scale = 0.02
        self.fold_shift = 0.01
        self.phase = phase
        self.encoder_dim = 128
        self.recon_decoder_dim = 32
        self.voxel_dim = voxel_dim
        basic_layer = TransformerBlock
        self.encoder = encoder(self.encoder_dim, basic_layer, scale)
        self.weight_generator = weight_generator(self.encoder_dim, 64)
        self.recon_net = recon_net(self.encoder_dim, self.recon_decoder_dim, basic_layer, scale)
        self.scale_fc = nn.Sequential(nn.Linear(self.encoder_dim, self.encoder_dim // 2),
                                 nn.ReLU(),
                                 nn.Linear(self.encoder_dim // 2, 1))  # 1-fold
        self.mlp = nn.Sequential(nn.Linear(self.encoder_dim, self.encoder_dim // 2),
                                 nn.ReLU(),
                                 nn.Linear(self.encoder_dim // 2, 3))  # 1-fold
        self.mlp2 = nn.Sequential(nn.Linear(self.encoder_dim, self.encoder_dim // 2),
                                 nn.ReLU(),
                                 nn.Linear(self.encoder_dim // 2, 9))  # 2-fold
        self.mlp3 = nn.Sequential(nn.Linear(self.encoder_dim, self.encoder_dim//2),
                                 nn.ReLU(),
                                 nn.Linear(self.encoder_dim//2, 12))  # 3-fold
        self.mlp_sphere = nn.Sequential(nn.Linear(self.encoder_dim, self.encoder_dim // 2),
                                  nn.ReLU(),
                                  nn.Linear(self.encoder_dim // 2, 7))  # 2-fold
        self.action = nn.Sequential(nn.Linear(self.encoder_dim, self.encoder_dim//2),
                                 nn.ReLU(),
                                 nn.Linear(self.encoder_dim//2, 3))  # action
        self.fc_position = nn.Sequential(
            nn.Linear(60, self.encoder_dim//2),
            nn.ReLU(),
            nn.Linear(self.encoder_dim//2, self.encoder_dim)
        )
        self.step_param = [0.25, 0.3]


    def local_plane_supervision(self, feature, clean_point, k_clean):
        plane_param = self.mlp(feature)  # add neighbor pos encoding
        orientation = plane_param[:, :, 0:3] / (
                torch.norm(plane_param[:, :, 0:3], dim=-1, keepdim=True) + 1e-6)
        # local supervision

        self_idx, g_dis = knn_point(k_clean, clean_point, clean_point, dis=True)
        g_dis = g_dis.max(-1)[0]
        self_neighbor = index_points(clean_point, self_idx)
        def to_surface_shift(self_neighbor, clean_point):
            max_shift = clean_point.unsqueeze(2) - self_neighbor
            final_shift =torch.matmul(max_shift, orientation.unsqueeze(3)) * orientation.unsqueeze(2)
            shift_dis = (final_shift**2).sum(-1).mean(-1)
            return shift_dis, final_shift
        ################# accuracy #######################
        B, N, _ = clean_point.shape
        shift_dis, _ = to_surface_shift(self_neighbor, clean_point)

        ################# completeness##############
        k_local = 32
        random_neighbor = (torch.rand(B, N, k_local, 3) - 0.5).to(clean_point.device) * 2 * (
                    g_dis.unsqueeze(2).unsqueeze(2) + 0.000) + clean_point.unsqueeze(2)
        _, shift = to_surface_shift(random_neighbor, clean_point)
        surface_point = random_neighbor + shift
        complete_dis = ((surface_point.unsqueeze(3) - self_neighbor.unsqueeze(2))**2).sum(-1).min(-1)[0].mean(-1)
        ############### average #############
        average_dis = (complete_dis/2 + shift_dis)* 5e3
        # local_chamfer, _ = chamfer_loss_chunk_local(shifted_local.reshape(-1, k_generate, 3),
        #                                               self_neighbor.reshape(-1, k_clean, 3), trainig=False)
        # np.savetxt("rand.txt", local_point.reshape(-1, k_clean, 3)[56].cpu().detach().numpy(), delimiter=";")
        # np.savetxt("shifted.txt", shifted_local.reshape(-1, k_clean, 3)[56].cpu().detach().numpy(), delimiter=";")
        # np.savetxt("gt.txt", self_neighbor.reshape(-1, k_clean, 3)[56].cpu().detach().numpy(), delimiter=";")
        # exit(0)
        return average_dis

    def local_2fold_supervision(self, feature, clean_point, k_clean):
        twofold_param = self.mlp2(feature)  # add neighbor pos encoding
        center_point = clean_point + F.tanh(twofold_param[:, :, 6:9]) * self.fold_shift
        orientation1 = twofold_param[:, :, 0:3] / (
                torch.norm(twofold_param[:, :, 0:3], dim=-1, keepdim=True) + 1e-6)
        orientation2 = twofold_param[:, :, 3:6] / (
                torch.norm(twofold_param[:, :, 3:6], dim=-1, keepdim=True) + 1e-6)
        outer = torch.cross(orientation1, orientation2)
        outer = outer / (torch.norm(outer, dim=-1, keepdim=True) + 1e-6)
        # local supervision
        self_idx, g_dis = knn_point(k_clean, clean_point, clean_point, dis=True)
        g_dis = g_dis.max(-1)[0]
        self_neighbor = index_points(clean_point, self_idx)

        #################
        B, N, _ = clean_point.shape
        # radius = 0.015
        # max_shift = (torch.rand(B, N, k_generate, 3) - 0.5).to(clean_point.device) * 2 * (
        #             g_dis.unsqueeze(2).unsqueeze(2) + 0.000)
        # max_dist = (max_shift ** 2).sum(-1)
        # local_point = center_point.unsqueeze(2) - max_shift

        def to_surface_shift(self_neighbor, clean_point):
            max_shift = center_point.unsqueeze(2) - self_neighbor
            local_point = self_neighbor  # clean_point.unsqueeze(2) - max_shift
            plane_shift = max_shift - torch.matmul(max_shift, outer.unsqueeze(3)) * outer.unsqueeze(2)
            proj1 = torch.matmul(plane_shift, orientation1.unsqueeze(3)) #* ori1.unsqueeze(0)
            proj2 = torch.matmul(plane_shift, orientation2.unsqueeze(3)) #* ori2.unsqueeze(0)
            shift1 = plane_shift - proj1* orientation1.unsqueeze(2)
            shift2 = plane_shift - proj2* orientation2.unsqueeze(2)
            ind1 = (0 >= proj1[:,:,:,0])
            ind2 = (0 >= proj2[:,:,:,0])
            final_shift = torch.ones_like(plane_shift) * plane_shift
            index_pp = ind1 & ind2
            B, M, K, _ = plane_shift.shape
            current_shit = torch.stack([shift1, shift2]).reshape(2, B*M*K, -1)
            dist_index = (current_shit ** 2).sum(-1).min(0)[1]
            final_shift[index_pp] = current_shit[dist_index, torch.arange(B*M*K)].reshape(B, M, K, -1)[index_pp]
            index_np = (~ind1) & ind2
            final_shift[index_np] = shift2[index_np]
            index_pn = (ind1) & (~ind2)
            final_shift[index_pn] = shift1[index_pn]
            shift_dis = (final_shift ** 2).sum(-1).mean(-1)
            return shift_dis, final_shift

        ################# accuaracy ##############
        shift_dis, _ = to_surface_shift(self_neighbor, clean_point)
        ################# completeness ##############
        k_local = 32
        random_neighbor = (torch.rand(B, N, k_local, 3) - 0.5).to(clean_point.device) * 2 * (
                g_dis.unsqueeze(2).unsqueeze(2) + 0.000) + clean_point.unsqueeze(2)
        _, shift = to_surface_shift(random_neighbor, clean_point)
        surface_point = random_neighbor + shift
        complete_dis = ((surface_point.unsqueeze(3) - self_neighbor.unsqueeze(2)) ** 2).sum(-1).min(-1)[0].mean(-1)
        ############### average #############
        average_dis = (complete_dis / 2 + shift_dis) * 5e3
        #
        # local_chamfer, _ = chamfer_loss_chunk_local(shifted_local.reshape(-1, k_generate, 3),
        #                                               self_neighbor.reshape(-1, k_clean, 3), trainig=False)
        # np.savetxt("rand.txt", local_point.reshape(-1, k_clean, 3)[56].cpu().detach().numpy(), delimiter=";")
        # np.savetxt("shifted.txt", shifted_local.reshape(-1, k_clean, 3)[56].cpu().detach().numpy(), delimiter=";")
        # np.savetxt("gt.txt", self_neighbor.reshape(-1, k_clean, 3)[56].cpu().detach().numpy(), delimiter=";")
        # exit(0)
        return average_dis

    def local_3fold_supervision(self, feature, clean_point, k_clean):
        three_fold_param = self.mlp3(feature)  # add neighbor pos encoding
        center_point = clean_point + F.tanh(three_fold_param[:, :, 0:3]) * self.fold_shift
        orientation1 = three_fold_param[:, :, 3:6] / (
        torch.norm(three_fold_param[:, :, 3:6], dim=-1, keepdim=True) + 1e-6)
        orientation2 = three_fold_param[:, :, 6:9] / (
        torch.norm(three_fold_param[:, :, 6:9], dim=-1, keepdim=True) + 1e-6)
        # orientation2 = orientation2 - torch.matmul(orientation1.unsqueeze(2), orientation2.unsqueeze(3))[:,:, 0]*orientation1
        # orientation2 = orientation2 / (
        #         torch.norm(orientation2, dim=-1, keepdim=True) + 1e-6)
        # orientation3 = torch.cross(orientation1, orientation2)
        orientation3 = three_fold_param[:, :, 9:12] / (
        torch.norm(three_fold_param[:, :, 9:12], dim=-1, keepdim=True) + 1e-6)
        # local supervision
        k_generate = 128
        self_idx, g_dis = knn_point(k_clean, clean_point, clean_point, dis=True)
        g_dis = g_dis.max(-1)[0]
        self_neighbor = index_points(clean_point, self_idx)

        #################
        B, N, _ = clean_point.shape
        # radius = 0.015
        def to_surface_shift(self_neighbor, clean_point):
            max_shift = center_point.unsqueeze(2) - self_neighbor
            local_point = self_neighbor
            # max_shift = (torch.rand(B, N, k_generate, 3) - 0.5).to(orientation3.device) * 2 * (
            #             g_dis.unsqueeze(2).unsqueeze(2) + 0.000)
            # max_dist = (max_shift ** 2).sum(-1)
            # local_point = center_point.unsqueeze(2) - max_shift

            outer12 = torch.cross(orientation1, orientation2)
            outer12 = outer12 / (torch.norm(outer12, dim=-1, keepdim=True) + 1e-6)
            shift12 = torch.matmul(max_shift, outer12.unsqueeze(3)) * outer12.unsqueeze(2)

            outer13 = torch.cross(orientation1, orientation3)
            outer13 = outer13 / (torch.norm(outer13, dim=-1, keepdim=True) + 1e-6)
            shift13 = torch.matmul(max_shift, outer13.unsqueeze(3)) * outer13.unsqueeze(2)

            outer23 = torch.cross(orientation2, orientation3)
            outer23 = outer23 / (torch.norm(outer23, dim=-1, keepdim=True) + 1e-6)
            shift23 = torch.matmul(max_shift, outer23.unsqueeze(3))* outer23.unsqueeze(2)

            proj12 = max_shift - shift12
            proj13 = max_shift - shift13
            proj23 = max_shift - shift23

            mu12 = (orientation1[:, :, 0] * orientation2[:, :, 1] - orientation2[:, :, 0] * orientation1[:, :, 1])
            mu12 = (mu12 + torch.sign(mu12) * 1e-7).unsqueeze(2)
            alpha12 = (proj12[:, :, :, 0] * orientation2[:, :, 1].unsqueeze(2) - orientation2[:, :, 0].unsqueeze(2) * proj12[:, :, :, 1]) / mu12
            beta12 = -(proj12[:, :, :, 0] * orientation1[:, :, 1].unsqueeze(2) - orientation1[:, :, 0].unsqueeze(2) * proj12[:, :, :, 1]) / mu12

            mu13 = (orientation1[:, :, 0] * orientation3[:, :,1] - orientation3[:, :, 0] * orientation1[:, :, 1])
            mu13 = (mu13 + torch.sign(mu13) * 1e-7).unsqueeze(2)
            alpha13 = (proj13[:, :, :, 0] * orientation3[:, :, 1].unsqueeze(2) - orientation3[:, :, 0].unsqueeze(2) * proj13[:, :, :, 1]) / mu13
            beta13 = -(proj13[:, :, :, 0] * orientation1[:, :, 1].unsqueeze(2) - orientation1[:, :, 0].unsqueeze(2) * proj13[:, :, :, 1]) / mu13

            mu23 = (orientation2[:, :, 0] * orientation3[:, :, 1] - orientation3[:, :, 0] * orientation2[:, :, 1])
            mu23 = (mu23 + torch.sign(mu23) * 1e-7).unsqueeze(2)
            alpha23 = (proj23[:, :, :, 0] * orientation3[:, :, 1].unsqueeze(2) - orientation3[:, :, 0].unsqueeze(2) * proj23[:, :, :, 1]) / mu23
            beta23 = -(proj23[:, :, :, 0] * orientation2[:, :, 1].unsqueeze(2) - orientation2[:, :, 0].unsqueeze(2) * proj23[:, :, :, 1]) / mu23

            final_dist = torch.ones_like(max_shift) * max_shift
            ind12 = (0 >= alpha12) & (0 >= beta12)
            ind13 = (0 >= alpha13) & (0 >= beta13)
            ind23 = (0 >= alpha23) & (0 >= beta23)
            # final_dist[ind12] = shift12[ind12]
            # final_dist[ind13] = shift13[ind13]
            # final_dist[ind23] = shift23[ind23]

            shift12[~ ind12] = max_shift[~ ind12]
            shift13[~ ind13] = max_shift[~ ind13]
            shift23[~ ind23] = max_shift[~ ind23]
            final_shift = torch.stack([shift12, shift13, shift23])
            B, M, K, C = max_shift.shape
            final_dist_index = (final_shift ** 2).sum(-1).reshape(3, -1).min(0)[1]
            final_dist = final_shift.reshape(3, B * M * K, C)[final_dist_index, torch.arange(B * M * K)].reshape(B, M, K,
                                                                                                                 -1)

            edge_index = ~(ind12 | ind13 | ind23)
            to_edge_or_center1 = \
                (max_shift + F.relu(- torch.matmul(max_shift, orientation1.unsqueeze(3))) * orientation1.unsqueeze(2))
            to_edge_or_center2 = \
                (max_shift + F.relu(- torch.matmul(max_shift, orientation2.unsqueeze(3))) * orientation2.unsqueeze(2))
            to_edge_or_center3 = \
                (max_shift + F.relu(-torch.matmul(max_shift, orientation3.unsqueeze(3))) * orientation3.unsqueeze(2))
            to_edge_or_center123 = torch.stack([to_edge_or_center1, to_edge_or_center2, to_edge_or_center3])
            min_index = (to_edge_or_center123 ** 2).sum(-1).reshape(3, -1).min(0)[1]
            to_edge_or_center = to_edge_or_center123.reshape(3, B * M * K, C)[min_index, torch.arange(B * M * K)].reshape(B,
                                                                                                                          M,
                                                                                                                          K,
                                                                                                                          -1)
            final_dist[edge_index] = to_edge_or_center[edge_index]
            shift_dis = (final_dist ** 2).sum(-1).mean(-1)
            return shift_dis, final_dist

        ################# accuaracy ##############
        shift_dis, _ = to_surface_shift(self_neighbor, clean_point)
        ################# completeness ##############
        k_local = 32
        random_neighbor = (torch.rand(B, N, k_local, 3) - 0.5).to(clean_point.device) * 2 * (
                g_dis.unsqueeze(2).unsqueeze(2) + 0.000) + clean_point.unsqueeze(2)
        _, shift = to_surface_shift(random_neighbor, clean_point)
        surface_point = random_neighbor + shift
        complete_dis = ((surface_point.unsqueeze(3) - self_neighbor.unsqueeze(2)) ** 2).sum(-1).min(-1)[0].mean(-1)
        ############### average #############
        average_dis = (complete_dis / 2 + shift_dis) * 5e3
        # local_chamfer, _ = chamfer_loss_chunk_local(shifted_local.reshape(-1, k_generate, 3),
        #                                               self_neighbor.reshape(-1, k_clean, 3), trainig=False)
        #
        # nc_dis = torch.matmul(orientation1.unsqueeze(2), orientation2.unsqueeze(3)) + \
        #          torch.matmul(orientation1.unsqueeze(2), orientation3.unsqueeze(3)) \
        #          + torch.matmul(orientation2.unsqueeze(2), orientation3.unsqueeze(3))
        # nc_dis = nc_dis.mean()
        # np.savetxt("rand.txt", random_neighbor[0, 56].cpu().detach().numpy(), delimiter=";")
        # np.savetxt("shifted.txt", surface_point[0, 56].cpu().detach().numpy(), delimiter=";")
        # # np.savetxt("gt.txt", self_neighbor.reshape(-1, k_clean, 3)[56].cpu().detach().numpy(), delimiter=";")
        # exit(0)
        return average_dis

    def local_sphere_supervision(self, feature, clean_point, k_clean = 12):
        sphere_param = self.mlp2(feature)  # add neighbor pos encoding
        center_point = clean_point + F.tanh(sphere_param[:, :, 3:6]) *self.fold_shift
        orientation1 = sphere_param[:, :, 0:3] / (
                torch.norm(sphere_param[:, :, 0:3], dim=-1, keepdim=True) + 1e-6)
        radius = F.sigmoid(sphere_param[:, :, 6])*0.01
        # local supervision
        self_idx, g_dis = knn_point(k_clean, clean_point, clean_point, dis=True)
        self_neighbor = index_points(clean_point, self_idx)
        #################
        B, N, _ = clean_point.shape
        # radius = 0.015
        # max_shift = (torch.rand(B, N, k_generate, 3) - 0.5).to(clean_point.device) * 2 * (
        #             g_dis.unsqueeze(2).unsqueeze(2) + 0.000)
        g_xyz = center_point.unsqueeze(2) - self_neighbor
        local_point = self_neighbor # clean_point.unsqueeze(2) - max_shift

        plane_shift = g_xyz - torch.matmul(g_xyz, orientation1.unsqueeze(3))* orientation1.unsqueeze(2)
        shift_norm = torch.norm(plane_shift, dim=-1, keepdim=True) + 1e-6
        shift_direction = plane_shift / shift_norm
        final_shift = (shift_norm - radius.unsqueeze(2).unsqueeze(2)) * shift_direction

        shifted_local = local_point + final_shift
        shift_dis = (final_shift**2).sum(-1).mean(-1)* 5e3
        # np.savetxt("rand.txt", local_point.reshape(-1, k_clean, 3)[56].cpu().detach().numpy(), delimiter=";")
        # np.savetxt("shifted.txt", shifted_local.reshape(-1, k_clean, 3)[56].cpu().detach().numpy(), delimiter=";")
        # np.savetxt("gt.txt", self_neighbor.reshape(-1, k_clean, 3)[56].cpu().detach().numpy(), delimiter=";")
        # exit(0)
        return shift_dis

    def cal_3fold_shift(self, feature, clean_point, detect_point, knn_num):
        three_fold_param = self.mlp3(feature)  # add neighbor pos encoding
        center_point = clean_point + F.tanh(three_fold_param[:,:,0:3])*self.fold_shift
        orientation1 = three_fold_param[:, :, 3:6] / (
                    torch.norm(three_fold_param[:, :, 3:6], dim=-1, keepdim=True) + 1e-6)

        # orientation2 = three_fold_param[:, :, 6:9] / (
        #         torch.norm(three_fold_param[:, :, 6:9], dim=-1, keepdim=True) + 1e-6)
        # orientation2 = orientation2 - torch.matmul(orientation1.unsqueeze(2), orientation2.unsqueeze(3))[:, :,
        #                               0] * orientation1
        # orientation2 = orientation2 / (
        #         torch.norm(orientation2, dim=-1, keepdim=True) + 1e-6)
        # orientation3 = torch.cross(orientation1, orientation2)
        orientation2 = three_fold_param[:, :, 6:9] / (
                    torch.norm(three_fold_param[:, :, 6:9], dim=-1, keepdim=True) + 1e-6)
        orientation3 = three_fold_param[:, :, 9:12] / (
                    torch.norm(three_fold_param[:, :, 9:12], dim=-1, keepdim=True) + 1e-6)

        # orientation2 = -torch.sign(torch.matmul(orientation1.unsqueeze(2), orientation2.unsqueeze(3))[:,:,0])*orientation2
        # orientation3 = -torch.sign(torch.matmul(orientation1.unsqueeze(2), orientation3.unsqueeze(3))[:, :, 0]) * orientation3

        # np.savetxt("to_center.txt", local_point[0,128].cpu().detach().numpy() , delimiter=";")
        # np.savetxt("shifted.txt", shifted[0,512].cpu().detach().numpy(), delimiter=";")
        # np.savetxt("rand.txt", local_point[0,512].cpu().detach().numpy(), delimiter=";")
        # exit(0)
        xxxxx = 0
        #### for visual test #####
        # iii = 36
        # num_local = 10000
        # chosen_point = center_point[0, iii]
        # ori1 = orientation1[0, iii]
        # ori2 = orientation2[0, iii]
        # ori3 = orientation3[0, iii]
        # # ori1 = torch.tensor([0.0, -1.0, 0.0]).to(orientation1.device)
        # # ori2 = torch.tensor([0.866, 0.5, 0]).to(orientation1.device)
        # # ori3 = torch.tensor([-0.866, 0.5, 0]).to(orientation1.device)
        # max_shift = (torch.rand(num_local,3)-0.5).to(ori1.device)
        #
        # max_dist = (max_shift**2).sum(-1)
        # local_point = chosen_point - max_shift
        #
        # outer12 = torch.cross(ori1, ori2)
        # outer12 = outer12/(torch.norm(outer12, dim=-1, keepdim=True) + 1e-6)
        # shift12 = torch.matmul(max_shift, outer12.unsqueeze(1))*outer12.unsqueeze(0)
        #
        # outer13 = torch.cross(ori1, ori3)
        # outer13 = outer13 / (torch.norm(outer13, dim=-1, keepdim=True) + 1e-6)
        # shift13 = torch.matmul(max_shift, outer13.unsqueeze(1)) * outer13.unsqueeze(0)
        #
        # outer23 = torch.cross(ori2, ori3)
        # outer23 = outer23 / (torch.norm(outer23, dim=-1, keepdim=True) + 1e-6)
        # shift23 = torch.matmul(max_shift, outer23.unsqueeze(1)) * outer23.unsqueeze(0)
        #
        # proj12 = max_shift - shift12
        # proj13 = max_shift - shift13
        # proj23 = max_shift - shift23
        #
        # mu12 = (ori1[0]*ori2[1] - ori2[ 0]*ori1[1]) + torch.sign(ori1[0]*ori2[1] - ori2[ 0]*ori1[1]) * 1e-7
        # alpha12 = (proj12[:, 0]*ori2[1] - ori2[0]*proj12[:, 1])/mu12
        # beta12 = -(proj12[:, 0] * ori1[1] - ori1[0] * proj12[:, 1]) / mu12
        #
        # mu13 = (ori1[0] * ori3[1] - ori3[0] * ori1[1]) + torch.sign(ori1[0] * ori3[1] - ori3[0] * ori1[1]) * 1e-7
        # alpha13 = (proj13[:, 0] * ori3[1] - ori3[0] * proj13[:, 1]) / mu13
        # beta13 = -(proj13[:, 0] * ori1[1] - ori1[0] * proj13[:, 1]) / mu13
        #
        # mu23 = (ori2[0] * ori3[1] - ori3[0] * ori2[1]) + torch.sign(ori2[0] * ori3[1] - ori3[0] * ori2[1]) * 1e-7
        # alpha23 = (proj23[:, 0] * ori3[1] - ori3[0] * proj23[:, 1]) /mu23
        # beta23 = -(proj23[:, 0] * ori2[1] - ori2[0] * proj23[:, 1]) / mu23
        #
        # # one_shift = torch.ones_like(shift12)*100
        # final_dist = torch.ones_like(max_shift)*max_shift
        # ind12 = (0 >= alpha12) & (0.0 >= beta12)
        # ind13 = (0.0 >= alpha13) & (0.0 >= beta13)
        # ind23 = (0.0 >= alpha23) & (0.0 >= beta23)
        # shift12[~ ind12] = max_shift[~ ind12]
        # shift13[~ ind13] = max_shift[~ ind13]
        # shift23[~ ind23] = max_shift[~ ind23]
        # final_shift = torch.stack([shift12, shift13, shift23])
        # final_dist_index = (final_shift**2).sum(-1).min(0)[1]
        # final_shift = final_shift[final_dist_index, torch.arange(num_local)]
        # # final_dist[ind13] = shift13[ind13]
        # # final_dist[ind23] = shift23[ind23]
        # edge_index = ~(ind12|ind13|ind23)
        # to_edge_or_center1 = \
        #     (max_shift + F.relu(- torch.matmul(max_shift, ori1.unsqueeze(1))) * ori1)+1e-7
        # to_edge_or_center2 = \
        #     (max_shift + F.relu(- torch.matmul(max_shift, ori2.unsqueeze(1))) * ori2)+2e-7
        # to_edge_or_center3 = \
        #     (max_shift + F.relu(-torch.matmul(max_shift, ori3.unsqueeze(1))) * ori3)+3e-7
        # to_edge_or_center123 = torch.stack([to_edge_or_center1, to_edge_or_center2, to_edge_or_center3])
        # min_index = (to_edge_or_center123**2).sum(-1).min(0)[1]
        # to_edge_or_center = to_edge_or_center123[min_index, torch.arange(num_local)]
        # final_shift[edge_index] = to_edge_or_center[edge_index]
        # # sum = edge_index.long().sum()
        # final_dist = final_shift
        # shifted = local_point + final_dist
        #
        # ###########################################################
        # # fig = plt.figure()
        # # ax = fig.add_subplot(111, projection='3d')
        # # plot_center = chosen_point.cpu().detach().numpy()
        # # o1 = ori1.cpu().detach().numpy()
        # # o2 = ori2.cpu().detach().numpy()
        # # o3 = ori3.cpu().detach().numpy()
        # # ax.quiver(plot_center[0], plot_center[1], plot_center[2], o1[0], o1[1], o1[2], color=(1, 0, 0, 0.5))
        # # ax.quiver(plot_center[0], plot_center[1], plot_center[2], o2[0], o2[1], o2[2], color=(1, 0, 0, 0.5))
        # # ax.quiver(plot_center[0], plot_center[1], plot_center[2], o3[0], o3[1], o3[2], color=(1, 0, 0, 0.5))
        # # ax.set_xlabel('X')
        # # ax.set_xlim3d(-1, 1)
        # # ax.set_ylabel('Y')
        # # ax.set_ylim3d(-1, 1)
        # # ax.set_zlabel('Z')
        # # ax.set_zlim3d(-1, 1)
        # # ax.grid()
        # # # ax = Axes3D(fig)
        # # # p = local_point.cpu().detach().numpy()
        # # # ax.scatter(p[:,0], p[:,1], p[:,2], c='g',marker='*')
        # # # p2 = shifted.cpu().detach().numpy()
        # # # ax.scatter(p2[:, 0], p2[:, 1], p2[:, 2], c='r', marker='+')
        # # plt.show()
        # np.savetxt("to_edge0.txt", local_point[edge_index & (min_index == 0)].cpu().detach().numpy(), delimiter=";")
        # np.savetxt("to_edge1.txt", local_point[edge_index & (min_index == 1)].cpu().detach().numpy(), delimiter=";")
        # np.savetxt("to_edge2.txt", local_point[edge_index & (min_index == 2)].cpu().detach().numpy(), delimiter=";")
        #
        # np.savetxt("ori1.txt", (chosen_point.cpu() + torch.linspace(0, 1.0, 100).unsqueeze(1) * ori1.cpu().unsqueeze(0)).detach().numpy(), delimiter=";")
        # np.savetxt("ori2.txt", (chosen_point.cpu() + torch.linspace(0, 1.0, 100).unsqueeze(1) * ori2.cpu().unsqueeze(
        #     0)).detach().numpy(), delimiter=";")
        # np.savetxt("ori3.txt", (chosen_point.cpu() + torch.linspace(0, 1.0, 100).unsqueeze(1) * ori3.cpu().unsqueeze(
        #     0)).detach().numpy(), delimiter=";")
        #
        # np.savetxt("shifted.txt", shifted.cpu().detach().numpy(), delimiter=";")
        # np.savetxt("rand.txt", local_point.cpu().detach().numpy(), delimiter=";")
        # exit(0)
        ###########################
        # orientation2 = torch.matmul(orientation1.unsqueeze(2), orientation2.unsqueeze(3))[:,:,0]*orientation2
        # orientation3 = torch.matmul(orientation1.unsqueeze(2), orientation3.unsqueeze(3))[:, :, 0] * orientation3
        point_index = knn_point(knn_num, clean_point, detect_point)
        g_ori1 = index_points(orientation1, point_index)
        g_ori2 = index_points(orientation2, point_index)
        g_ori3 = index_points(orientation3, point_index)
        g_xyz = index_points(center_point, point_index)

        B, M, k, _ = g_xyz.shape
        max_shift = g_xyz - detect_point.unsqueeze(2)

        outer12 = torch.cross(g_ori1, g_ori2)
        outer12 = outer12/(torch.norm(outer12, dim=-1, keepdim=True) + 1e-6)
        shift12 = torch.matmul(max_shift.unsqueeze(3), outer12.unsqueeze(4))[:,:,:,0]*outer12

        outer13 = torch.cross(g_ori1, g_ori3)
        outer13 = outer13 / (torch.norm(outer13, dim=-1, keepdim=True) + 1e-6)
        shift13 = torch.matmul(max_shift.unsqueeze(3), outer13.unsqueeze(4))[:,:,:,0] * outer13

        outer23 = torch.cross(g_ori2, g_ori3)
        outer23 = outer23 / (torch.norm(outer23, dim=-1, keepdim=True) + 1e-6)
        shift23 = torch.matmul(max_shift.unsqueeze(3), outer23.unsqueeze(4))[:,:,:,0] * outer23

        proj12 = max_shift - shift12
        proj13 = max_shift - shift13
        proj23 = max_shift - shift23

        mu12 = (g_ori1[:,:,:,0]*g_ori2[:,:,:,1] - g_ori2[:,:,:,0]*g_ori1[:,:,:,1])
        mu12 = mu12 + torch.sign(mu12)*1e-7
        alpha12 = (proj12[:,:,:,0]*g_ori2[:,:,:,1]- g_ori2[:,:,:,0]*proj12[:,:,:,1])/mu12
        beta12 = -(proj12[:,:,:,0] * g_ori1[:,:,:,1] - g_ori1[:,:,:,0] * proj12[:,:,:,1]) / mu12

        mu13 = (g_ori1[:,:,:,0] * g_ori3[:,:,:,1] - g_ori3[:,:,:,0] * g_ori1[:,:,:,1])
        mu13 = mu13 + torch.sign(mu13) * 1e-7
        alpha13 = (proj13[:,:,:,0] * g_ori3[:,:,:,1] - g_ori3[:,:,:,0] * proj13[:,:,:,1]) / mu13
        beta13 = -(proj13[:,:,:,0] * g_ori1[:,:,:,1] - g_ori1[:,:,:,0] * proj13[:,:,:,1]) / mu13

        mu23 = (g_ori2[:,:,:,0] * g_ori3[:,:,:,1] - g_ori3[:,:,:,0] * g_ori2[:,:,:,1])
        mu23 = mu23 + torch.sign(mu23) * 1e-7
        alpha23 = (proj23[:,:,:,0] * g_ori3[:,:,:,1] - g_ori3[:,:,:,0] * proj23[:,:,:,1]) /mu23
        beta23 = -(proj23[:,:,:,0] * g_ori2[:,:,:,1] - g_ori2[:,:,:,0] * proj23[:,:,:,1]) / mu23

        final_dist = torch.ones_like(max_shift) * max_shift
        ind12 = (0 >= alpha12) & (0 >= beta12)
        ind13 = (0 >= alpha13) & (0 >= beta13)
        ind23 = (0 >= alpha23) & (0 >= beta23)
        shift12[~ ind12] = max_shift[~ ind12]
        shift13[~ ind13] = max_shift[~ ind13]
        shift23[~ ind23] = max_shift[~ ind23]
        final_shift = torch.stack([shift12, shift13, shift23])
        B, M, K, C = max_shift.shape
        final_dist_index = (final_shift ** 2).sum(-1).reshape(3, -1).min(0)[1]
        final_dist = final_shift.reshape(3, B*M*K, C)[final_dist_index, torch.arange(B*M*K)].reshape(B,M,K, -1)

        edge_index = ~(ind12 | ind13 | ind23)
        to_edge_or_center1 = \
            (max_shift + F.relu(- torch.matmul(max_shift.unsqueeze(3), g_ori1.unsqueeze(4)))[:,:,:,0] * g_ori1)
        to_edge_or_center2 = \
            (max_shift + F.relu(- torch.matmul(max_shift.unsqueeze(3), g_ori2.unsqueeze(4)))[:,:,:,0] * g_ori2)
        to_edge_or_center3 = \
            (max_shift + F.relu(-torch.matmul(max_shift.unsqueeze(3), g_ori3.unsqueeze(4)))[:,:,:,0] * g_ori3)
        to_edge_or_center123 = torch.stack([to_edge_or_center1, to_edge_or_center2, to_edge_or_center3])
        min_index = (to_edge_or_center123 ** 2).sum(-1).reshape(3, -1).min(0)[1]
        to_edge_or_center = to_edge_or_center123.reshape(3, B*M*K, C)[min_index, torch.arange(B*M*K)].reshape(B,M,K, -1)
        final_dist[edge_index] = to_edge_or_center[edge_index]

        # distance = torch.norm(max_shift, dim=-1)
        # weight = torch.softmax(-distance * 10000, -1)
        # shift = torch.matmul(weight.unsqueeze(2), final_dist)[:,:,0,:]
        return 0, final_dist

    def cal_2fold_shift(self, feature, clean_point, detect_point, knn_num):
        twofold_param = self.mlp2(feature)  # add neighbor pos encoding
        center_point = clean_point + F.tanh(twofold_param[:, :, 6:9]) *self.fold_shift
        orientation1 = twofold_param[:, :, 0:3] / (
                torch.norm(twofold_param[:, :, 0:3], dim=-1, keepdim=True) + 1e-6)
        orientation2 = twofold_param[:, :, 3:6] / (
                torch.norm(twofold_param[:, :, 3:6], dim=-1, keepdim=True) + 1e-6)
        outer = torch.cross(orientation1, orientation2)
        outer = outer / (torch.norm(outer, dim=-1, keepdim=True) + 1e-6)

        #### for visual
        # iii = 244
        # num_local = 10000
        # chosen_point = center_point[0, iii]
        # ori1 = orientation1[0, iii]
        # ori2 = orientation2[0, iii]
        # out = outer[0,iii]
        # # ori1 = torch.tensor([0.0, -1.0, 0.0]).to(orientation1.device)
        # # ori2 = torch.tensor([0.866, 0.5, 0]).to(orientation1.device)
        # # ori3 = torch.tensor([-0.866, 0.5, 0]).to(orientation1.device)
        # max_shift = (torch.rand(num_local,3)-0.5).to(ori1.device)
        #
        # max_dist = (max_shift**2).sum(-1)
        # local_point = chosen_point - max_shift
        # # self_n_idx = knn_point(knn_num, clean_point, clean_point)
        # # self_neighbor = index_points(clean_point, self_n_idx)
        # plane_shift = max_shift - torch.matmul(max_shift, out.unsqueeze(1)) * out.unsqueeze(0)
        # proj1 = torch.matmul(plane_shift, ori1.unsqueeze(1)) #* ori1.unsqueeze(0)
        # proj2 = torch.matmul(plane_shift, ori2.unsqueeze(1)) #* ori2.unsqueeze(0)
        # shift1 = plane_shift - proj1* ori1.unsqueeze(0)
        # shift2 = plane_shift - proj2* ori2.unsqueeze(0)
        # ind1 = (0 >= proj1[:,0])
        # ind2 = (0 >= proj2[:,0])
        # final_shift = torch.ones_like(plane_shift) * plane_shift
        # index_pp = ind1 & ind2
        # K, _ = plane_shift.shape
        # current_shit = torch.stack([shift1, shift2]).reshape(2, K, -1)
        # dist_index = (current_shit ** 2).sum(-1).min(0)[1]
        # final_shift[index_pp] = current_shit[dist_index, torch.arange(K)].reshape(K, -1)[index_pp]
        # index_np = (~ind1) & ind2
        # final_shift[index_np] = shift2[index_np]
        # index_pn = (ind1) & (~ind2)
        # final_shift[index_pn] = shift1[index_pn]
        #
        # shifted = local_point + final_shift
        #
        # ###########################################################
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # plot_center = chosen_point.cpu().detach().numpy()
        # o1 = ori1.cpu().detach().numpy()
        # o2 = ori2.cpu().detach().numpy()
        # # o3 = ori3.cpu().detach().numpy()
        # ax.quiver(plot_center[0], plot_center[1], plot_center[2], o1[0], o1[1], o1[2], color=(1, 0, 0, 0.5))
        # ax.quiver(plot_center[0], plot_center[1], plot_center[2], o2[0], o2[1], o2[2], color=(1, 0, 0, 0.5))
        # # ax.quiver(plot_center[0], plot_center[1], plot_center[2], o3[0], o3[1], o3[2], color=(1, 0, 0, 0.5))
        # ax.set_xlabel('X')
        # ax.set_xlim3d(-1, 1)
        # ax.set_ylabel('Y')
        # ax.set_ylim3d(-1, 1)
        # ax.set_zlabel('Z')
        # ax.set_zlim3d(-1, 1)
        # ax.grid()
        # # ax = Axes3D(fig)
        # # p = local_point.cpu().detach().numpy()
        # # ax.scatter(p[:,0], p[:,1], p[:,2], c='g',marker='*')
        # # p2 = shifted.cpu().detach().numpy()
        # # ax.scatter(p2[:, 0], p2[:, 1], p2[:, 2], c='r', marker='+')
        # plt.show()
        # # np.savetxt("to_edge0.txt", local_point[edge_index & (min_index == 0)].cpu().detach().numpy(), delimiter=";")
        # # np.savetxt("to_edge1.txt", local_point[edge_index & (min_index == 1)].cpu().detach().numpy(), delimiter=";")
        # # np.savetxt("to_edge2.txt", local_point[edge_index & (min_index == 2)].cpu().detach().numpy(), delimiter=";")
        #
        # np.savetxt("ori1.txt", (chosen_point.cpu() + torch.linspace(0, 1.0, 100).unsqueeze(1) * ori1.cpu().unsqueeze(0)).detach().numpy(), delimiter=";")
        # np.savetxt("ori2.txt", (chosen_point.cpu() + torch.linspace(0, 1.0, 100).unsqueeze(1) * ori2.cpu().unsqueeze(
        #     0)).detach().numpy(), delimiter=";")
        # # np.savetxt("ori3.txt", (chosen_point.cpu() + torch.linspace(0, 1.0, 100).unsqueeze(1) * ori3.cpu().unsqueeze(
        # #     0)).detach().numpy(), delimiter=";")
        #
        # np.savetxt("shifted.txt", shifted.cpu().detach().numpy(), delimiter=";")
        # np.savetxt("rand.txt", local_point.cpu().detach().numpy(), delimiter=";")
        # exit(0)

        ###############

        point_index = knn_point(knn_num, clean_point, detect_point)
        # weight, point_index = self.weight_generator(feature, clean_point, detect_point, knn_num=knn_num)  # * scale.unsqueeze(2)
        g_ori1 = index_points(orientation1, point_index)
        g_ori2 = index_points(orientation2, point_index)
        g_outer = index_points(outer, point_index)
        g_xyz = index_points(center_point, point_index)
        g_xyz = g_xyz - detect_point.unsqueeze(2)
        plane_shift = g_xyz - torch.matmul(g_xyz.unsqueeze(3), g_outer.unsqueeze(4))[:,:,:,0]*g_outer
        proj1 = torch.matmul(g_ori1.unsqueeze(3), plane_shift.unsqueeze(4))[:,:,:,0]#*g_ori1
        proj2 = torch.matmul(g_ori2.unsqueeze(3), plane_shift.unsqueeze(4))[:,:,:,0]#*g_ori2
        shift1 = plane_shift - proj1*g_ori1
        shift2 = plane_shift - proj2*g_ori2
        ind1 = (0>=proj1[:,:,:,0])
        ind2 = (0>=proj2[:,:,:,0])
        final_shift = torch.ones_like(plane_shift)*plane_shift
        index_pp = ind1&ind2
        B, M, K, _ = plane_shift.shape
        current_shit = torch.stack([shift1, shift2]).reshape(2, B*M*K, -1)
        dist_index = (current_shit**2).sum(-1).min(0)[1]
        final_shift[index_pp] = current_shit[dist_index, torch.arange(B*M*K)].reshape(B, M, K, -1)[index_pp]
        index_np = (~ind1)&ind2
        final_shift[index_np] = shift2[index_np]
        index_pn = (ind1) &(~ind2)
        final_shift[index_pn] = shift1[index_pn]
        # shift = torch.matmul(weight.unsqueeze(2), final_shift)[:, :, 0, :]
        return 0, final_shift

    def cal_plane_shift(self, feature, clean_point, detect_point, knn_num):
        plane_param = self.mlp(feature)  # add neighbor pos encoding
        orientation = plane_param[:, :, 0:3] / (
                torch.norm(plane_param[:, :, 0:3], dim=-1, keepdim=True) + 1e-6)
        # weight, point_index = self.weight_generator(feature, clean_point, detect_point, knn_num=knn_num)  # * scale.unsqueeze(2)
        point_index = knn_point(knn_num, clean_point, detect_point)
        g_ori = index_points(orientation, point_index)
        g_xyz = index_points(clean_point, point_index)
        g_xyz = g_xyz - detect_point.unsqueeze(2)
        final_shift = (torch.matmul(g_xyz.unsqueeze(3), g_ori.unsqueeze(4))[:,:,:,0]* g_ori)
        # shift = (final_shift* weight.unsqueeze(3)).sum(-2)
        return 0, final_shift

    def cal_sphere_shift(self, feature, clean_point, detect_point, knn_num):
        sphere_param = self.mlp2(feature)  # add neighbor pos encoding
        center_point = clean_point + F.tanh(sphere_param[:, :, 3:6]) *self.fold_shift
        orientation1 = sphere_param[:, :, 0:3] / (
                torch.norm(sphere_param[:, :, 0:3], dim=-1, keepdim=True) + 1e-6)
        radius = F.sigmoid(sphere_param[:, :, 6])*0.01
        point_index = knn_point(knn_num, clean_point, detect_point)
        # weight, point_index = self.weight_generator(feature, clean_point, detect_point, knn_num=knn_num)  # * scale.unsqueeze(2)
        g_ori1 = index_points(orientation1, point_index)
        g_radius = index_points(radius.unsqueeze(2), point_index)
        g_center = index_points(center_point, point_index)
        g_xyz = g_center - detect_point.unsqueeze(2)
        plane_shift = g_xyz - torch.matmul(g_xyz.unsqueeze(3), g_ori1.unsqueeze(4))[:,:,:,0]*g_ori1
        shift_norm = torch.norm(plane_shift, dim=-1, keepdim=True) + 1e-6
        shift_direction = plane_shift/shift_norm
        final_shift = (shift_norm - g_radius)*shift_direction
        return 0, final_shift

    def mean_shift_error(self, feature, clean_point):
        B, N, _ = clean_point.shape
        # action = self.action(feature) =
        # action = F.softmax(action, -1)
        k_clean = 32
        # action = F.softmax(action, -1)
        shift_plane_err = self.local_plane_supervision(feature, clean_point, k_clean)
        return shift_plane_err

    def cal_shift(self, feature, clean_point, detect_point, error, knn_num=16, training=True):

        _, shift_plane = self.cal_plane_shift(feature, clean_point, detect_point, knn_num)
        B, M, K,  _ = shift_plane.shape
        point_index = knn_point(knn_num, clean_point, detect_point)
        weight = self.weight_generator(feature, clean_point, detect_point, error, point_index,
                                                    knn_num=knn_num)  # * scale.unsqueeze(2)
        # _, shift_chose = self.cal_2fold_shift(feature, clean_point, detect_point)
        shift = (shift_plane * weight.unsqueeze(3)).sum(-2)
        # shift_weight = (shift_plane* weight.unsqueeze(3)).sum(-2)
        return shift

    def cal_pc(self, detect_point, coding, clean_point, error, refine=True):
        ###############################move#########
        chunk_size = 100000
        chunk_num = int(math.ceil(detect_point.shape[1] / chunk_size))
        query_list = torch.chunk(detect_point, chunk_num, dim=1)
        logit_list = []
        for q in query_list:
            # logit_uniform = self.recon_net.decoder_far(*(coding + (clean_point, q))) * self.noise_scale
            shift = self.cal_shift(coding[0], clean_point, q, error)
            # shift = self.cal_shift(orientation, clean_point, q+shift0, coding[0])
            # shift = shift1 + shift0
            logit_list.append(shift)
        logit_list = torch.cat(logit_list, 1)
        shift_pc = detect_point + logit_list

        ############################## refinement ######################################
        if refine:
            chunk_num = int(math.ceil(shift_pc.shape[1] / chunk_size))
            query_list = torch.chunk(shift_pc, chunk_num, dim=1)
            logit_list_refine = []
            for q in query_list:
                shift = self.recon_net.decoder_far(*(coding + (clean_point, q))) * 0.01
                logit_list_refine.append(shift)
            logit_list_refine = torch.cat(logit_list_refine, 0)
            shifted_pc_refine = shift_pc + logit_list_refine
            shift = logit_list_refine + logit_list
        else:
            ##############
            shifted_pc_refine = shift_pc
            shift = logit_list
        return shifted_pc_refine, shift

    def cal_pc_v0(self, detect_point, coding, clean_point, error):
        ###############################shrink###
        # point_index = knn_point(4, clean_point, detect_point)
        # g_xyz = index_points(clean_point, point_index)
        # shift0 = (g_xyz - detect_point.unsqueeze(2)).mean(2)
        # detect_point = detect_point + shift0
        # np.savetxt("sample.txt", detect_point[0].detach().cpu().numpy(), delimiter=";")
        # np.savetxt("cl.txt", clean_point[0].detach().cpu().numpy(), delimiter=";")
        # exit(0)
        ###############################
        shift = self.cal_shift(coding[0], clean_point, detect_point, error)
        # shift = self.cal_shift(orientation, clean_point, detect_point+shift0, coding[0])
        # shift = shift1 + shift0
        shifted_xyz = shift + detect_point
        # ###############################################
        # shift = self.recon_net.decoder_far(*(coding + (clean_point, detect_point))) * self.noise_scale / 2
        # shifted_xyz = detect_point + shift
        shift_refine = self.recon_net.decoder_far(*(coding + (clean_point, shifted_xyz.detach())))*0.01
        # shift_refine = torch.clamp(shift_refine, max=self.noise_scale/2, min=-self.noise_scale/2)
        shifted_xyz_refine = shift_refine + shifted_xyz.detach()
        shift = shift + shift_refine
        # shifted_xyz_refine = shifted_xyz
        return shifted_xyz, shifted_xyz_refine, shift

    def network_loss_orientation(self, clean_point, points_chamfer, scale, epoch=0):
        # points_chamfer = torch.cat([clean_point, points_chamfer], 1)
        points_chamfer = clean_point
        ###############################chamfer#########################
        B = clean_point.shape[0]
        coding = self.encoder(clean_point)
        point_scale = self.scale_fc(coding[0])
        error = self.mean_shift_error(coding[0], clean_point)
        local_chamfer = error.mean()
        repeat_num = 1
        repeat_point1 = clean_point.unsqueeze(2).repeat(1, 1, repeat_num, 1)
        repeat_point2 = clean_point.unsqueeze(2).repeat(1, 1, repeat_num, 1)
        noise1 = (torch.rand_like(repeat_point1) - 0.5) * 2 * self.noise_scale
        noise2 = (torch.rand_like(repeat_point2) - 0.5) * 2 * self.noise_scale

        far_point = (repeat_point1 + noise1).reshape(B, -1, 3)
        close_point = (repeat_point2 + noise2).reshape(B, -1, 3)
        detect_point = torch.cat([far_point, close_point], 1)
        N_close_start = 0  # far_point.shape[1]
        shifted_xyz, shifted_xyz_refine, shift = self.cal_pc_v0(detect_point, coding, clean_point, point_scale)
        chamfer, _ = chamfer_loss_chunk_no_grad_closest(detect_point, shifted_xyz, points_chamfer, scale)
        chamfer_refine, _ = chamfer_loss_chunk_no_grad_closest(detect_point, shifted_xyz_refine, points_chamfer, scale)
        shift_norm = torch.norm(shift, p=2, dim=-1, keepdim=True)
        L_norm = shift_norm.mean()*0
        ########################shift along gradient###########################
        alpha = 0.25+0.6*np.random.random()
        new_detect_point = detect_point[:, N_close_start:, :] + alpha* shift[:, N_close_start:, :]
        new_shifted_xyz, new_shifted_xyz_refine, new_shift = self.cal_pc_v0(new_detect_point, coding, clean_point, point_scale)
        L = (new_shift/(1-alpha) - shift).abs().sum(-1).mean()*200

        # vanish_point = torch.cat([clean_point, shifted_xyz_refine], 1)
        _, _, vanish_shift = self.cal_pc_v0(clean_point, coding, clean_point, point_scale)
        vanish_norm = torch.norm(vanish_shift, p=2, dim=-1).mean()*100

        return chamfer, vanish_norm, L_norm, chamfer_refine, L, L_norm*0, local_chamfer


    def forward_iou_ori(self, clean_point, points_chamfer, query_point, grid_tensor, sign=None, scale=1):
        # self.noise_scale = 0.00
        B = clean_point.shape[0]
        # fps_idx1 = farthest_point_sample(clean_point, 1000).long()  # [B, npoi
        coding = self.encoder(clean_point)
        point_scale = self.scale_fc(coding[0])
        # clean_point_embed = get_3d_sincos_pos_embed_from_point(60, clean_point)
        # clean_point_embed = self.fc_position(clean_point_embed)
        # local_index = knn_point(4, clean_point, clean_point)
        # p2p_feature = index_points(coding[0], local_index).mean(2)
        # orientation = self.mlp(coding[0])
        # orientation = orientation / (torch.norm(orientation, dim=-1, keepdim=True) + 1e-6)
        error = self.mean_shift_error(coding[0], clean_point)
        detect_point = clean_point.unsqueeze(2).repeat(1, 1, 100, 1)
        noise = (torch.rand_like(detect_point) - 0.5) * 2 * self.noise_scale
        detect_point = detect_point + noise
        detect_point = detect_point.reshape(B, -1, 3)
        shifted_pc_refine, logit_list0 = self.cal_pc(detect_point, coding, clean_point, point_scale)
        shifted_pc_refine, logit_list1 = self.cal_pc(shifted_pc_refine, coding, clean_point, point_scale)
        logit_list = logit_list0  + logit_list1
        # udf = torch.norm(logit_list, p=2, dim=-1).squeeze()
        # index = torch.where((udf > 0.005) & (udf < 0.015))

        # shifted_pc_refine = shifted_pc_refine[0][index].unsqueeze(0)
        # shifted_pc_refine, _ = self.cal_pc(shifted_pc_refine, coding, clean_point, orientation)
        # chamfer, _ = chamfer_loss_chunk_no_grad(shifted_pc_refine, points_chamfer, trainig=False)
        shifted_pc_ref=ine = shifted_pc_refine[0].cpu().detach().numpy()

        shifted_pc_refine2, logit_list0 = self.cal_pc(query_point, coding, clean_point, point_scale)
        shifted_pc_refine2, logit_list1 = self.cal_pc(shifted_pc_refine2, coding, clean_point, point_scale)
        # shifted_pc_refine2, logit_list2 = self.cal_pc(shifted_pc_refine2, coding, clean_point, orientation)
        logit_list = logit_list0 + logit_list1 # + logit_list2
        udf = torch.norm(logit_list, p=2, dim=-1).squeeze()
        udf = udf.cpu().numpy()
        tree = KDTree(clean_point[0].cpu().numpy())
        dist, idx = tree.query(query_point[0].cpu().numpy())
        # udf = np.stack([dist, udf]).min(0)
        near_index = np.where(dist<self.noise_scale)[0]
        udf = np.clip(udf*scale, a_max=self.noise_scale, a_min=0)
        logit_list = logit_list*scale

        # chosen_idx = np.where(udf < self.noise_scale*10000)
        if sign is not None:
            sign = np.sign(sign[grid_tensor[:, 0], grid_tensor[:, 1], grid_tensor[:, 2]])
            udf = -(udf * sign.reshape(-1))

        result = self.noise_scale * np.ones(4 * self.voxel_dim ** 3)  # .scatter_(1, index, src)
        result = result.reshape(4, self.voxel_dim, self.voxel_dim, self.voxel_dim)
        generate_orientation = np.concatenate(
            [np.expand_dims(udf, axis=1), logit_list.squeeze().cpu().numpy()],
            -1).squeeze()[near_index].squeeze()
        grid_tensor = grid_tensor.numpy()
        result[:, grid_tensor[near_index, 0], grid_tensor[near_index, 1], grid_tensor[near_index, 2]] = generate_orientation.transpose()
        return shifted_pc_refine*scale, 0, detect_point*scale, result

    # def forward_iou_ori(self, clean_point, points_chamfer, query_point, grid_tensor, sign=None, scale=1):
    #     # self.noise_scale = 0.00
    #     B = clean_point.shape[0]
    #     # fps_idx1 = farthest_point_sample(clean_point, 1000).long()  # [B, npoi
    #     coding = self.encoder(clean_point)
    #     point_scale = self.scale_fc(coding[0])
    #     # clean_point_embed = get_3d_sincos_pos_embed_from_point(60, clean_point)
    #     # clean_point_embed = self.fc_position(clean_point_embed)
    #     # local_index = knn_point(4, clean_point, clean_point)
    #     # p2p_feature = index_points(coding[0], local_index).mean(2)
    #     # orientation = self.mlp(coding[0])
    #     # orientation = orientation / (torch.norm(orientation, dim=-1, keepdim=True) + 1e-6)
    #     error, action = self.mean_shift_error(coding[0], clean_point)
    #     detect_point = clean_point.unsqueeze(2).repeat(1, 1, 100, 1)
    #     noise = (torch.rand_like(detect_point) - 0.5) * 2 * self.noise_scale
    #     detect_point = detect_point + noise
    #     detect_point = detect_point.reshape(B, -1, 3)
    #     shifted_pc_refine, logit_list0 = self.cal_pc(detect_point, coding, clean_point, action, point_scale)
    #     shifted_pc_refine, logit_list1 = self.cal_pc(shifted_pc_refine, coding, clean_point, action, point_scale)
    #     logit_list = logit_list0  + logit_list1
    #     # udf = torch.norm(logit_list, p=2, dim=-1).squeeze()
    #     # index = torch.where((udf > 0.005) & (udf < 0.015))
    #
    #     # shifted_pc_refine = shifted_pc_refine[0][index].unsqueeze(0)
    #     # shifted_pc_refine, _ = self.cal_pc(shifted_pc_refine, coding, clean_point, orientation)
    #     # chamfer, _ = chamfer_loss_chunk_no_grad(shifted_pc_refine, points_chamfer, trainig=False)
    #     shifted_pc_ref=ine = shifted_pc_refine[0].cpu().detach().numpy()
    #
    #     shifted_pc_refine2, logit_list0 = self.cal_pc(query_point, coding, clean_point, action, point_scale)
    #     shifted_pc_refine2, logit_list1 = self.cal_pc(shifted_pc_refine2, coding, clean_point, action, point_scale)
    #     # shifted_pc_refine2, logit_list2 = self.cal_pc(shifted_pc_refine2, coding, clean_point, orientation)
    #     logit_list = logit_list0 + logit_list1 # + logit_list2
    #     udf = torch.norm(logit_list, p=2, dim=-1).squeeze()
    #     udf = udf.cpu().numpy()
    #     # tree = KDTree(shifted_pc_refine)
    #     # dist, idx = tree.query(query_point[0].cpu().numpy())
    #     # udf = np.stack([dist, udf]).min(0)*scale
    #     udf = np.clip(udf*scale, a_max=self.noise_scale, a_min=0)
    #     logit_list = logit_list*scale
    #
    #     # chosen_idx = np.where(udf < self.noise_scale*10000)
    #     if sign is not None:
    #         sign = np.sign(sign[grid_tensor[:, 0], grid_tensor[:, 1], grid_tensor[:, 2]])
    #         udf = -(udf * sign.reshape(-1))
    #
    #     result = self.noise_scale * np.ones(4 * self.voxel_dim ** 3)  # .scatter_(1, index, src)
    #     result = result.reshape(4, self.voxel_dim, self.voxel_dim, self.voxel_dim)
    #     generate_orientation = np.concatenate(
    #         [np.expand_dims(udf, axis=1), logit_list.squeeze().cpu().numpy()],
    #         -1).squeeze()
    #     grid_tensor = grid_tensor.numpy()
    #     result[:, grid_tensor[:, 0], grid_tensor[:, 1], grid_tensor[:, 2]] = generate_orientation.transpose()
    #     return shifted_pc_refine*scale, 0, detect_point*scale, result
    #     # return result


    def forward_iou_cal(self, clean_point, points_chamfer, query_point, grid_tensor, sign=None, scale=1):
        # self.noise_scale = 0.00
        B = clean_point.shape[0]
        # fps_idx1 = farthest_point_sample(clean_point, 1000).long()  # [B, npoi
        coding = self.encoder(clean_point)
        orientation = self.mlp(coding[0])
        orientation = orientation / (torch.norm(orientation, dim=-1, keepdim=True) + 1e-6)

        detect_point = clean_point.unsqueeze(2).repeat(1, 1, 100, 1)
        noise = (torch.rand_like(detect_point) - 0.5) * 2 * self.noise_scale
        detect_point = detect_point + noise
        detect_point = detect_point.reshape(B, -1, 3)
        shifted_pc_refine, logit_list0 = self.cal_pc(detect_point, coding, clean_point, orientation)
        shifted_pc_refine, logit_list1 = self.cal_pc(shifted_pc_refine, coding, clean_point, orientation)
        logit_list = logit_list0 + logit_list1  # + logit_list2
        udf = torch.norm(logit_list, p=2, dim=-1).squeeze()
        index = torch.where((udf > -0.005) & (udf < 0.15))
        direction = logit_list / (udf.unsqueeze(0).unsqueeze(2) + 1e-7)
        direction = direction[0][index].unsqueeze(0)
        shifted_pc_refine = shifted_pc_refine[0][index].unsqueeze(0)
        chamfer, _ = chamfer_loss_chunk_no_grad(shifted_pc_refine, points_chamfer, trainig=False)

        tree = KDTree(shifted_pc_refine[0].cpu().numpy())
        dist, idx = tree.query(query_point[0].cpu().numpy(), k=16)
        weight = F.softmax(torch.from_numpy(-dist*10))
        relative_xyz = index_points(shifted_pc_refine, torch.from_numpy(idx.astype(np.int64)).unsqueeze(0)) - query_point.unsqueeze(2)
        g_ori = index_points(direction, torch.from_numpy(idx.astype(np.int64)).unsqueeze(0))
        shift = torch.matmul(relative_xyz.unsqueeze(3), g_ori.unsqueeze(4)).squeeze().unsqueeze(0).unsqueeze(3)
        logit_list = (shift.cpu() * g_ori.cpu() * (weight.unsqueeze(0).unsqueeze(3))).sum(-2)
        shifted_pc_refine = (query_point[0].cpu() + logit_list).unsqueeze(0).to(query_point.device)
        # shifted_pc_refine, _ = self.cal_pc(shifted_pc_refine, coding, clean_point, orientation)

        # shifted_pc_refine = shifted_pc_refine[0].cpu().detach().numpy()
        #
        # shifted_pc_refine2, logit_list0 = self.cal_pc(query_point, coding, clean_point, orientation)
        # shifted_pc_refine2, logit_list1 = self.cal_pc(shifted_pc_refine2, coding, clean_point, orientation)
        # # shifted_pc_refine2, logit_list2 = self.cal_pc(shifted_pc_refine2, coding, clean_point, orientation)
        # logit_list = logit_list0 + logit_list1 # + logit_list2
        udf = torch.norm(logit_list, p=2, dim=-1).squeeze()
        udf = udf.cpu().numpy()
        # tree = KDTree(shifted_pc_refine)
        # dist, idx = tree.query(query_point[0].cpu().numpy())
        # udf = np.stack([dist, udf]).min(0)*scale
        udf = np.clip(udf*scale, a_max=self.noise_scale, a_min=0)
        logit_list = logit_list*scale

        # chosen_idx = np.where(udf < self.noise_scale*10000)
        if sign is not None:
            # sign = np.sign(sign[grid_tensor[chosen_idx, 0], grid_tensor[chosen_idx, 1], grid_tensor[chosen_idx, 2]])
            udf = -(udf * sign.reshape(-1))

        result = self.noise_scale * np.ones(4 * self.voxel_dim ** 3)  # .scatter_(1, index, src)
        result = result.reshape(4, self.voxel_dim, self.voxel_dim, self.voxel_dim)
        generate_orientation = np.concatenate(
            [np.expand_dims(udf, axis=1), logit_list.squeeze().cpu().numpy()],
            -1).squeeze()
        grid_tensor = grid_tensor.numpy()
        result[:, grid_tensor[:, 0], grid_tensor[:, 1], grid_tensor[:, 2]] = generate_orientation.transpose()
        shifted_pc_refine = shifted_pc_refine[0].cpu().numpy()
        return shifted_pc_refine*scale, chamfer*scale, detect_point*scale, result, orientation
        # return result

    def forward_g(self, clean_point, points_chamfer, query_point, grid_tensor, sign=None, scale=1.0):
        chamfer, _ = chamfer_loss_chunk_no_grad_closest(points_chamfer, points_chamfer, clean_point, scale)
        tree = KDTree(points_chamfer[0].cpu().numpy())
        dist, idx = tree.query(query_point[0].cpu().numpy())
        udf = dist
        logit_list = torch.from_numpy(points_chamfer[0].cpu().numpy()[idx] - query_point[0].cpu().numpy()).unsqueeze(0)
        chosen_idx = np.where(udf < self.noise_scale)
        if sign is not None:
            sign = np.sign(sign[grid_tensor[chosen_idx, 0], grid_tensor[chosen_idx, 1], grid_tensor[chosen_idx, 2]])
            udf = (udf[chosen_idx] * sign)[0]
        else:
            udf = (udf[chosen_idx])
        result = self.noise_scale * np.ones(4 * self.voxel_dim ** 3)  # .scatter_(1, index, src)
        result = result.reshape(4, self.voxel_dim, self.voxel_dim, self.voxel_dim)
        generate_orientation = np.concatenate(
            [np.expand_dims(udf, axis=1), logit_list.squeeze().cpu().numpy()[chosen_idx]],
            -1).squeeze()
        grid_tensor = grid_tensor.numpy()
        result[:, grid_tensor[chosen_idx, 0][0], grid_tensor[chosen_idx, 1][0],
        grid_tensor[chosen_idx, 2][0]] = generate_orientation.transpose()
        return result, chamfer


    def forward_near(self, clean_point, near_point):
        coding = self.encoder(clean_point)
        orientation = self.mlp(coding[0])
        orientation = orientation / (torch.norm(orientation, dim=-1, keepdim=True) + 1e-6)

        shifted_pc_refine2, logit_list0 = self.cal_pc(near_point, coding, clean_point, orientation)
        shifted_pc_refine2, logit_list1 = self.cal_pc(shifted_pc_refine2, coding, clean_point, orientation)
        # shifted_pc_refine2, logit_list2 = self.cal_pc(shifted_pc_refine2, coding, clean_point, orientation)
        logit_list = logit_list0 + logit_list1 # + logit_list2
        udf = torch.norm(logit_list, p=2, dim=-1).squeeze()
        udf = udf.detach().cpu().numpy()
        udf = np.clip(udf, a_max=self.noise_scale, a_min=0)
        logit_list = logit_list
        return udf, logit_list



    def forward_pc(self, clean_point, points_chamfer):
        # self.noise_scale = 0.00
        B = clean_point.shape[0]
        # fps_idx1 = farthest_point_sample(clean_point, 1000).long()  # [B, npoi
        coding = self.encoder(clean_point)
        orientation = self.mlp(coding[0])
        orientation = orientation / (torch.norm(orientation, dim=-1, keepdim=True) + 1e-6)

        detect_point = clean_point.unsqueeze(2).repeat(1, 1, 33, 1)
        noise = (torch.rand_like(detect_point) - 0.5) * 2 * self.noise_scale
        detect_point = detect_point + noise
        detect_point = detect_point.reshape(B, -1, 3)
        shifted_pc_refine, _ = self.cal_pc(detect_point, coding, clean_point, orientation)
        shifted_pc_refine, _ = self.cal_pc(shifted_pc_refine, coding, clean_point, orientation)
        # shifted_pc_refine, _ = self.cal_pc(shifted_pc_refine, coding, clean_point, orientation)
        chamfer, _ = chamfer_loss_chunk_no_grad(shifted_pc_refine, points_chamfer, trainig=False)
        return chamfer
