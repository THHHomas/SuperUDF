import os
import time
import math
import random
import matplotlib.pyplot as plt

import numpy
import numpy as np
import h5py
from tqdm import tqdm
import open3d as o3d
from torch.optim.lr_scheduler import CosineAnnealingLR
import trimesh
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from attention_layer2 import bsp_network
from tensorboardX import SummaryWriter
from torchstat import stat
from torch import optim
from torch.autograd import Variable
from thop import profile
import mcubes
# from bspt import digest_bsp, get_mesh, get_mesh_watertight
# from bspt_slow import digest_bsp, get_mesh, get_mesh_watertight

from utils import *
from pointconv_util import knn_point, index_points, draw_direction
from scipy import ndimage

voxel_dim = 256
import torch
import torch.nn.functional as F


def SoftCrossEntropy(inputs, target, reduction='average'):
    log_likelihood = -F.log_softmax(inputs, dim=1)
    batch = inputs.shape[0]
    if reduction == 'average':
        loss = torch.sum(torch.mul(log_likelihood, target)) / batch
    else:
        loss = torch.sum(torch.mul(log_likelihood, target))
    return loss


def generate_softlabel(batch_voxels):
    result = []
    for idx in range(batch_voxels.shape[0]):
        voxel1= ndimage.binary_dilation((batch_voxels[idx,0]), structure=ndimage.generate_binary_structure(3,3), iterations=1).astype(np.float)*0.5
        voxel2 = ndimage.binary_dilation((batch_voxels[idx, 0]), structure=ndimage.generate_binary_structure(3, 3),
                                         iterations=2).astype(np.float)*0.25
        voxel3 = ndimage.binary_dilation((batch_voxels[idx, 0]), structure=ndimage.generate_binary_structure(3, 3),
                                         iterations=3).astype(np.float) * 0.125
        stacked_voxel = np.stack([batch_voxels[idx, 0], voxel1, voxel2, voxel3]).max(0)
        result.append(stacked_voxel)
    batch_voxels = np.expand_dims(np.stack(result), axis=1)
    return batch_voxels


def knn_point_with_dis(nsample, xyz, new_xyz, raduis=0.15):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz[:, :, 0:3], xyz[:, :, 0:3]).squeeze()
    # normaldists = square_distance(new_xyz[:, :, 3:], xyz[:, :, 3:]).squeeze()
    # d1, d1_dix = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=True)
    # k=2000
    # d2 = normaldists[k, d1_dix[k]]
    # r, t = d1.mean(), d2.mean()
    # dist = 10*sqrdists + 0*normaldists
    group_dis, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=True)
    return group_dis, group_idx


def position_encode(x, dim):
    x = x.unsqueeze(1)
    B = x.shape[0]
    feature = torch.arange(0, dim//2, device=x.device).float().unsqueeze(0)
    sin_feature = torch.sin(x/10000**(2*feature/168.0))
    cos_feature = torch.cos(x / 10000 ** (2 * feature / 168.0))
    feature = torch.stack([sin_feature, cos_feature]).permute(1,2,0).reshape(B, -1)
    return feature




class BSP_AE(object):
    def __init__(self, config):
        """
		Args:
			too lazy to explain
		"""
        self.config = config
        self.phase = config.phase

        # progressive training
        # 1-- (16, 16*16*16)
        # 2-- (32, 16*16*16)
        # 3-- (64, 16*16*16*4)
        self.sample_vox_size = config.sample_vox_size
        if self.sample_vox_size == 16:
            self.load_point_batch_size = 16 * 16 * 16
        elif self.sample_vox_size == 32:
            self.load_point_batch_size = 16 * 16 * 16
        elif self.sample_vox_size == 64:
            self.load_point_batch_size = 16 * 16 * 16 * 4
        self.shape_batch_size = 2
        self.point_batch_size = 16 * 16 * 16
        self.input_size = 64  # input voxel grid size

        self.ef_dim = 32

        self.dataset_name = config.dataset
        self.dataset_load = self.dataset_name + '_train'
        if not (config.train):
            self.dataset_load = self.dataset_name + '_test'
        self.checkpoint_dir = config.checkpoint_dir
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.dataset_name)
        self.data_dir = config.data_dir
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device('cpu')

        # build model
        # self.denoise_net = denoise_net()
        # self.denoise_net.to(self.device)
        # self.load(task='denoise')

        self.bsp_network = bsp_network(self.phase, voxel_dim=voxel_dim)
        self.bsp_network.to(self.device)
        self.optimizer = torch.optim.Adam(self.bsp_network.parameters(), lr=config.learning_rate,
                                          betas=(config.beta1, 0.999))

        # pytorch does not have a checkpoint manager
        # have to define it myself to manage max num of checkpoints to keep
        self.max_to_keep = 2
        self.checkpoint_path = os.path.join(self.checkpoint_dir, self.dataset_name)
        self.checkpoint_name = 'BSP_AE.model'
        self.checkpoint_manager_list = [None] * self.max_to_keep
        self.checkpoint_manager_pointer = 0


    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.config.learning_rate * (0.3 ** (epoch // 60))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def load(self, task=''):
        # load previous checkpoint
        checkpoint_txt = str(os.path.join(self.checkpoint_path, "checkpoint"+task))
        print(checkpoint_txt, os.path.exists(checkpoint_txt), type(checkpoint_txt))
        if os.path.exists(checkpoint_txt):
            fin = open(checkpoint_txt)
            model_dir = fin.readline().strip()
            fin.close()
            if task == 'denoise':
                self.denoise_net.load_state_dict(torch.load(model_dir), strict=True)
                print(" [*] Load Denoise SUCCESS")
            else:
                state  = torch.load(model_dir)
                self.bsp_network.load_state_dict(torch.load(model_dir), strict=True)
                print(" [*] Load SUCCESS")

            return True
        else:
            print(" [!] Load failed...")
            return False

    def save(self, epoch, task=''):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        save_dir = os.path.join(self.checkpoint_path,
                                task + str(self.sample_vox_size) + "-" + str(self.phase) + "-" + str(
                                    epoch) + "_ori.pth")
        self.checkpoint_manager_pointer = (self.checkpoint_manager_pointer + 1) % self.max_to_keep
        # delete checkpoint
        if self.checkpoint_manager_list[self.checkpoint_manager_pointer] is not None:
            if os.path.exists(self.checkpoint_manager_list[self.checkpoint_manager_pointer]):
                os.remove(self.checkpoint_manager_list[self.checkpoint_manager_pointer])
        # save checkpoint
        torch.save(self.bsp_network.state_dict(), save_dir)
        # update checkpoint manager
        self.checkpoint_manager_list[self.checkpoint_manager_pointer] = save_dir
        # write file
        checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint"+task)
        fout = open(checkpoint_txt, 'w')
        for i in range(self.max_to_keep):
            pointer = (self.checkpoint_manager_pointer + self.max_to_keep - i) % self.max_to_keep
            if self.checkpoint_manager_list[pointer] is not None:
                fout.write(self.checkpoint_manager_list[pointer] + "\n")
        fout.close()

    def train(self, config):
        # load previous checkpoint
        cls = self.config.dataset.split("/")[3].split("_")[1]
        writer = SummaryWriter("log/"+cls+"/pt")
        # self.load()

        training_epoch = 100
        start_time = time.time()
        assert config.epoch == 0 or config.iteration == 0
        scheduler = CosineAnnealingLR(self.optimizer, training_epoch, eta_min=config.learning_rate*0.1)
        self.bsp_network.train()
        point_cloud, point_cloud2, sign = load_data(train_file="data/test.h5")

        for epoch in range(0, training_epoch):
            scheduler.step()
            avg_loss_sp = 0
            avg_loss_tt = 0
            avg_loss_chamfer = 0
            avg_loss_chamfer_fine = 0
            avg_loss_norm = 0
            avg_loss_z = 0
            avg_near_nc = 0
            avg_near_near = 0
            avg_num = 0
            batch_num = point_cloud.shape[0]//self.shape_batch_size
            avg_num = 0
            shuf_list = torch.randperm(point_cloud.shape[0])
            for idx in tqdm(range(batch_num)):
                data_idx = shuf_list[idx * self.shape_batch_size:(idx + 1) * self.shape_batch_size]
                pc, pc2 = point_cloud[data_idx], point_cloud2[data_idx]
                pc, pc2 = torch.from_numpy(pc), torch.from_numpy(pc2)
                pc, pc2 = pc[:,:,0:3], pc2[:,:,0:3]
                # random_padding = 1.0  # (np.random.random()-0.5)*0.25 + 1

                # b_scale = pc.max(1)[0] - pc.min(1)[0]
                # scale = ((b_scale[:, 0] * b_scale[:, 1] + b_scale[:, 0] * b_scale[:, 2] + b_scale[:, 2]
                #           * b_scale[:, 1]) /1.2).sqrt().unsqueeze(1).unsqueeze(1)
                #
                # scale = torch.clamp(scale, min=1.0).to(self.device)

                # scale = scale / scale
                scale = torch.ones((2,1,1)).float().to(self.device)
                pc = pc.to(self.device)/scale
                pc2 = pc2.to(self.device)/scale
                self.bsp_network.zero_grad()

                chamfer, vanish_norm, L_near, chamfer_fine, norm, vertical_norm, normal_consistency = self.bsp_network.network_loss_orientation(
                    pc, pc2, scale, epoch=epoch)
                loss = chamfer * 4 + vanish_norm + chamfer_fine * 0.5 + norm + vertical_norm + L_near + normal_consistency
                loss.backward()
                self.optimizer.step()
                avg_loss_sp += loss.item()
                avg_loss_tt += vanish_norm.item()
                avg_loss_chamfer += chamfer.item()
                avg_loss_chamfer_fine += chamfer_fine.item()
                avg_loss_norm += norm.item()
                avg_loss_z += vertical_norm.item()
                avg_near_near += L_near.item()
                avg_near_nc += normal_consistency.item()
                avg_num += 1

            writer.add_scalar("loss", avg_loss_sp / avg_num, global_step=epoch)
            writer.add_scalar("acc", avg_loss_tt / avg_num, global_step=epoch)
            print(str(
                self.sample_vox_size) + " Epoch: [%2d/%2d] time: %4.4f, loss: %.6f, vanish_norm: %.6f, chamfer: %.6f, chamfer_fine: %.6f, norm: %.6f, vertical norm:%.6f, nc:%.6f, near:%.6f" % (
                      epoch, training_epoch, time.time() - start_time, avg_loss_sp / avg_num, avg_loss_tt / avg_num
                      , avg_loss_chamfer / avg_num, avg_loss_chamfer_fine / avg_num, avg_loss_norm / avg_num,
                      avg_loss_z / avg_num, avg_near_nc / avg_num, avg_near_near / avg_num))
            # if epoch%10==9:
            # 	self.test_1(config,"train_"+str(self.sample_vox_size)+"_"+str(epoch))
            if epoch % 10 == 9:
                self.save(epoch)

        self.save(training_epoch)


    def test_dae3(self, config):
        # load previous checkpoint
        if not self.load(): exit(-1)
        sign = None
        # for m in self.bsp_network.modules():
        #     if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
        #         m.running_mean = None
        #         m.running_var = None

        self.bsp_network.eval()
        acc_mean = 0
        total_num = 0
        padding = 1.0
        acc_list = []
        point_cloud, point_cloud2, sign = load_data(train_file="data/test.h5")
        # point_cloud = np.expand_dims(np.loadtxt("5_inputpc.txt", delimiter=";"), axis=0).astype(np.float32)
        # point_cloud2 = point_cloud
        data_num = min(point_cloud.shape[0], config.end) - config.start

        with torch.no_grad():
            for t in range(config.start, config.start + data_num):

                pc = torch.from_numpy(point_cloud[t]).to(self.device).unsqueeze(0)/padding
                pc2 = torch.from_numpy(point_cloud2[t]).to(self.device).unsqueeze(0)/padding
                pc = pc[:, :, 0:3]
                pc2 = pc2[:, :, 0:3]
                if sign is not None:
                    c_sign = sign[t]
                else:
                    c_sign = None
                B = 1
                expand_index = []
                expand_point = []
                pc = pc + 0.5
                for idx in range(B):
                    box = (torch.cat([pc.max(dim=1)[0]+0.05,
                           pc.min(dim=1)[0]-0.05], 1))
                    box = torch.clamp((box*voxel_dim).round().long(), min=0, max=voxel_dim).squeeze()
                    grid = torch.meshgrid([torch.arange(box[3], box[0]), torch.arange(box[4], box[1]), torch.arange(box[5], box[2])])
                    # label = batch_voxels[grid[0].reshape(-1), grid[1].reshape(-1), grid[2].reshape(-1)]
                    # label = torch.zeros(test_dim**3, dtype=torch.long)# batch_voxels[idx, 0, grid[0].reshape(-1), grid[1].reshape(-1), grid[2].reshape(-1)].long()
                    grid = torch.stack(grid).reshape(3, -1).permute(1, 0)
                    grid_point = grid.cuda().float()/voxel_dim # + 0.5/float(voxel_dim)
                    expand_point.append(grid_point)
                    expand_index.append(grid)
                expand_point = torch.stack(expand_point)
                expand_point = expand_point - 0.5
                pc = pc - 0.5

                scale = torch.ones(1).to(self.device) * 1.0
                # b_scale = clean_point.max(1)[0] - clean_point.min(1)[0]
                # scale = ((b_scale[:, 0] * b_scale[:, 1] + b_scale[:, 0] * b_scale[:, 2] +
                #           b_scale[:, 2] * b_scale[:, 1]) / 2 * 4).sqrt().unsqueeze(1).unsqueeze(1)
                # scale = torch.clamp(scale, min=1.0)
                b_scale = pc.max(1)[0] - pc.min(1)[0]
                scale = ((b_scale[:, 0] * b_scale[:, 1] + b_scale[:, 0] * b_scale[:, 2] + b_scale[:, 2]
                          * b_scale[:, 1]) / 1.2).sqrt().unsqueeze(1).unsqueeze(1)

                scale = torch.clamp(scale, min=1.0).to(self.device)
                scale = scale/scale
                expand_point = expand_point / scale
                pc = pc / scale
                pc2 = pc2 / scale

                # shift_pc, chamfer_loss, noise_pc, udf = self.bsp_network.forward_iou_ori(pc, pc2, expand_point, grid, c_sign)
                shift_pc, chamfer_loss, noise_pc, udf = self.bsp_network.forward_iou_ori(pc, pc2,
                                                                                         expand_point, grid,
                                                                                         c_sign,
                                                                                         float(scale.cpu().numpy()))
                # pc = pc*padding
                # pc2 = pc2*padding
                # shift_pc = shift_pc*padding
                # noise_pc = noise_pc*padding
                #######################################################################
                clamp = udf[0, voxel_dim//2 ]
                plt.imshow(np.abs(clamp), cmap=plt.cm.jet)
                plt.savefig("/home/magician/Pictures/screen/x"+str(t)+".png")
                #
                # clamp = udf[0, :, 60]
                # plt.imshow(np.abs(clamp), cmap=plt.cm.jet)
                # plt.savefig("/home/magician/Pictures/screen/y" + str(t) + ".png")
                #
                # clamp = udf[0, :, :, 60]
                # plt.imshow(np.abs(clamp), cmap=plt.cm.jet)
                # plt.savefig("/home/magician/Pictures/screen/z" + str(t) + ".png")
                ##########################################################
                # detect_point_list = torch.chunk(expand_point, 200000, dim=1)
                # pred_point_list = torch.chunk(pred, 200000, dim=0)
                # net_out_all_list = []
                # for idx, d in enumerate(detect_point_list):
                #     pred_pc = d[0][torch.where(pred_point_list[idx] > 0.5)].cpu().numpy()
                #     net_out_all_list.append(pred_pc)
                # output = np.concatenate(net_out_all_list, 0)
                #
                # np.savetxt(config.sample_dir + "/" + self.txt_data[t] + "_predpc.txt",
                #            output, delimiter=";")
                total_num += 1
                #
                #
                # # mesh = extract_mesh_tsdf(net_out_all, padding=padding)
                # # trimesh.smoothing.filter_laplacian(mesh, iterations=1)
                # # mesh.export(config.sample_dir + "/" + self.txt_data[t] + ".off")
                if t > 50:
                    np.save(config.sample_dir + "/udf_data/train/" + str(t) + "_udf.npy",
                            udf)
                else:
                    if not os.path.exists(config.sample_dir + "/udf_data/test/"):
                        os.makedirs(config.sample_dir + "/udf_data/test/")
                    np.save(config.sample_dir + "/udf_data/test/" + str(t) + "_udf.npy",
                            udf)
                np.savetxt(config.sample_dir + "/" + str(t) + "_chamfer_pc.txt",
                          (pc2[0]).cpu().detach().numpy(), delimiter=";")
                np.savetxt(config.sample_dir + "/" + str(t) + "_shiftpc.txt",
                           (shift_pc[0].cpu().detach().numpy()), delimiter=";")
                np.savetxt(config.sample_dir + "/" + str(t) + "_inputpc.txt",
                           (pc[0]).cpu().detach().numpy(), delimiter=";")

                # print("[sample%d]"%t, chamfer_loss.item())
                # acc_list.append(chamfer_loss.item())
                print(t)
        index = torch.topk(torch.tensor(acc_list), k=min(10, data_num), largest=False)[1]
        # for idx in index:
        #     print(self.txt_data[idx])
        print("box_predict acc is： ", acc_mean/total_num)


def smooth_process(batch_voxels):
    batch_voxels_ = F.pad(batch_voxels, (1, 1, 1, 1, 1, 1), 'constant', 0)
    result = torch.zeros_like(batch_voxels)
    for x_shift in range(0,3):
        for y_shift in range(0, 3):
            for z_shift in range(0, 3):
                result += batch_voxels_[x_shift:x_shift+64, y_shift:y_shift+64, z_shift:z_shift+64]
    batch_voxels[torch.where(result == 1)] = 0
    batch_voxels[torch.where(result // 16 == 1)] = 1
    return batch_voxels


def extract_mesh_tsdf(voxel, padding=1.0):
    if isinstance(voxel, numpy.ndarray):
        voxel = torch.from_numpy(voxel)
    voxel = voxel.squeeze()
    n_x, n_y, n_z = voxel.shape
    voxel = F.pad(voxel, (1, 1, 1, 1, 1, 1), 'constant', -100)
    voxel = voxel.cpu().detach().numpy()

    vertices, triangles = mcubes.marching_cubes(voxel, math.log(0.2))
    vertices -= 1
    vertices /= np.array([n_x, n_y, n_z])
    vertices -= 0.5
    vertices *= padding
    # # Undo padding
    # matrix = rotation(-math.pi / 2).numpy()
    # vertices = np.matmul(vertices, matrix)
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    return mesh


def extract_mesh_voxel(voxel):
    if isinstance(voxel, numpy.ndarray):
        voxel = torch.from_numpy(voxel)
    voxel = voxel.squeeze()
    n_x, n_y, n_z = voxel.shape
    voxel = F.pad(voxel, (1, 1, 1, 1, 1, 1), 'constant', 0)
    voxel = voxel.cpu().detach().numpy()

    vertices, triangles = mcubes.marching_cubes(voxel, 0.5)
    vertices -= 1
    vertices /= np.array([n_x, n_y, n_z])
    vertices -= 0.5
    # # Undo padding
    # matrix = rotation(-math.pi / 2).numpy()
    # vertices = np.matmul(vertices, matrix)
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    return mesh
