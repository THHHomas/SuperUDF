import math
import open3d as o3d
from libkdtree import KDTree
import os
from transformer import get_1d_sincos_pos_embed_from_grid
import torch
import mcubes
import time
import trimesh
from tqdm import tqdm
import numpy as np
import h5py
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformer import TransformerBlock, upsample_layer_batch
from transformer_pos_enc import get_3d_sincos_pos_embed_from_point
from udf_extractor import generate_ver_tri
from ud_network import vertex_net2, pair_net, transformer_net, vertex_net, dual_vertex_net

data_width = 5
grid_dim = -1


def filter_far(vertices, triangles, pc):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.1, 32))
    normal = np.asarray(pcd.normals)

    triangle_vertices = vertices[triangles.reshape(-1).astype(np.int32), :].reshape(-1, 3, 3)

    ####################dis#######################
    triangles_center = triangle_vertices.mean(-2).astype(np.float32)

    tree = KDTree(pc)
    nearest_dis, index_nearest = tree.query(triangles_center)
    normal_nearest = normal[index_nearest]
    pc_nearest = pc[index_nearest]
    shift = triangles_center - pc_nearest
    vertical_shift = shift - np.matmul(np.expand_dims(shift, axis=1), np.expand_dims(normal_nearest, axis=2))[:, 0]*normal_nearest
    vertical_norm = np.linalg.norm(vertical_shift, ord=2, axis=1)
    index = np.where(vertical_norm<0.02)
    triangles = triangles[index]
    return vertices, triangles


def data_collection(data_list, result_file, strip=2, train=False):
    data_collected = []
    label_collected = []
    shift_collected = []
    for p in tqdm(sorted(data_list)):
        data_total = np.load(p)
        if data_total.shape[0] == 4:
            data = data_total[0]
        else:
            data = data_total

        # data = np.expand_dims(data, axis=0)
        dim = data.shape
        global grid_dim
        grid_dim = dim[0]

        abs_data = np.abs(data[:, :, :])
        sign_data = data[:, :, :]

        ratio = []
        for i_idx in range(0, dim[0] - data_width, strip):
            for j_idx in range(0, dim[1] - data_width, strip):
                for k_idx in range(0, dim[2] - data_width, strip):
                    if train and np.random.random() > 1:
                        continue
                    # field = data_total[1:, i_idx:i_idx + data_width, j_idx:j_idx + data_width,
                    #         k_idx:k_idx + data_width]
                    current_grid = abs_data[i_idx:i_idx + data_width, j_idx:j_idx + data_width,
                                   k_idx:k_idx + data_width]
                    # current_grid = np.concatenate([np.expand_dims(current_grid, axis=0), field], 0)
                    # min_udf = abs_data[i_idx+data_width//2-1:i_idx +data_width//2+2:2,
                    #           j_idx+data_width//2-1:j_idx+data_width//2+2:2, k_idx+data_width//2-1:k_idx +data_width//2+2:2].min()
                    min_udf = np.abs(current_grid[0]).min()
                    max_udf = np.abs(current_grid[0]).max()
                    if min_udf > 0.015 or max_udf > 0.25:
                        continue
                    # grid = np.concatenate([np.expand_dims(current_grid, axis=0), data[1:,i_idx:i_idx + data_width, j_idx:j_idx + data_width, k_idx:k_idx + data_width]], 0)
                    # shift = np.expand_dims(np.array([i_idx, j_idx, k_idx]).astype(np.float32) / 2.0, axis=0)
                    shift = np.expand_dims(np.array([i_idx, j_idx, k_idx]).astype(np.float32) / float(strip), axis=0)
                    label = np.sign(sign_data[i_idx + data_width // 2 - 1:i_idx + data_width // 2 + 2:2,
                                    j_idx + data_width // 2 - 1:j_idx + data_width // 2 + 2:2,
                                    k_idx + data_width // 2 - 1:k_idx + data_width // 2 + 2:2]).reshape(-1)
                    # label = np.sign(sign_data[i_idx + data_width // 2:i_idx + data_width // 2 + 2,
                    #                 j_idx + data_width // 2:j_idx + data_width // 2 + 2,
                    #                 k_idx + data_width // 2:k_idx + data_width // 2 + 2]).reshape(-1)

                    shift_collected.append(shift)
                    label_collected.append(label)
                    data_collected.append(current_grid)

    data_collected = np.stack(data_collected)
    label_collected = np.stack(label_collected)
    shift_collected = np.stack(shift_collected)

    with h5py.File(result_file, 'w') as f:
        f.create_dataset("data", data=data_collected)
        f.create_dataset("label", data=label_collected)
        f.create_dataset("shift", data=shift_collected)
    return data_collected, label_collected


def generate_data_list(cls, train=False):
    # root_data_dir = "../ShapeNet"
    # data_dir = "./samples/bsp_ae_out/udf_data/train/"
    data_dir = "data/npy/"
    # data_dir = "/disk2/unsupervised reconstruction/samples/bsp_ae_out/04530566/train"
    # data_dir = "/disk2/occupancy_networks-master/data/ShapeNet.build/04530566/2_binvox/"
    data_list = []
    for x in os.listdir(data_dir):
        if x[-3:] == "npy":
            data_list.append(os.path.join(data_dir, x))
    return data_list[0:20]


def data_generation(cls):
    train_list = generate_data_list(cls, train=False)
    data_collection(train_list, "data/train_vessel256.h5", train=True)
    # test_list = generate_data_list(cls, train=False)


def data_load(path, data_width=5):
    with h5py.File(path, 'r') as f:
        data = f["data"][:].astype(np.float32)
        shift = f["shift"][:].astype(np.float32)
        label = f["label"][:].astype(np.int64)
        # data_width = 5
        # data_shape = data.shape[-1]
        # data = data[:, data_shape//2-data_width//2:data_shape//2+data_width//2+1,
        #        data_shape//2-data_width//2:data_shape//2+data_width//2+1,
        #        data_shape//2-data_width//2:data_shape//2+data_width//2+1]
    return data, label, shift


def train():
    epochs = 200
    network = vertex_net(128)
    network.to("cuda")
    # network.load_state_dict(torch.load("data/checkpoint_aug.pth"), strict=True)

    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3,
                                 betas=(0.5, 0.999), weight_decay=0)
    scheduler = CosineAnnealingLR(optimizer, epochs, eta_min=1e-3)
    network.train()
    batchsize = 2048
    data, label, _ = data_load("data/train_vessel256.h5", data_width=data_width)

    for epoch in range(epochs):
        scheduler.step()
        loss_ob = 0
        acc_ob = 0
        shuf_list = torch.randperm(data.shape[0])
        iter_num = min(data.shape[0] // batchsize, 30000)
        for idx in tqdm(range(iter_num)):
            data_idx = shuf_list[idx * batchsize:(idx + 1) * batchsize]
            batch_data, batch_label = data[data_idx], label[data_idx]
            batch_data = torch.from_numpy(batch_data).to("cuda")
            batch_label = torch.from_numpy(batch_label).to("cuda")
            batch_data = batch_data.unsqueeze(1)
            ############ augmentation ##############
            # # # random scale
            # batch_data[:, 0] = batch_data[:, 0] / batch_data[:, 0].reshape(-1, data_width ** 3).mean(-1).unsqueeze(
            #     1).unsqueeze(1).unsqueeze(1)
            # scale = np.random.random() * 1 + 0.5
            # batch_data = batch_data * scale
            # # random flip
            flip_dim = int(np.random.random() * 4) % 4
            batch_label = batch_label.reshape(batchsize, 2, 2, 2)
            if flip_dim < 3:
                batch_data = torch.flip(batch_data, (flip_dim + 2,))
                batch_label = torch.flip(batch_label, (flip_dim + 1,))
            # random rotation
            rot_dim = int(np.random.random() * 4) % 4
            if rot_dim < 3:
                batch_data = torch.rot90(batch_data, 1, dims=[rot_dim + 2, (rot_dim + 1) % 3 + 2])
                batch_label = torch.rot90(batch_label, 1, dims=[rot_dim + 1, (rot_dim + 1) % 3 + 1])
            batch_label = batch_label.reshape(batchsize, -1)
            ########################################
            batch_label = batch_label*batch_label[:, 0].unsqueeze(1)
            out = network(batch_data)
            loss, acc = network.cal_loss(out, batch_label)
            network.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ob += loss.item()
            acc_ob += acc.item()
            if epoch % 10 == 9:
                torch.save(network.state_dict(), "data/checkpoint_aug.pth")

        print("Epoch:%d, loss:%f, acc:%f" % (epoch, loss_ob / (idx + 1), acc_ob / (idx + 1)))
        # torch.save(network.state_dict(), "data/checkpoint.pth")


def test():
    if data_width % 2 == 1:
        strip = 2
    else:
        strip = 1
    construct = True
    predict_path = "data/npy/"
    # predict_path = "/disk2/unsupervised reconstruction/samples/bsp_ae_out/udf_data/train/"  # calculate point shift
    # predict_path = "/disk2/unsupervised reconstruction/samples/bsp_ae_out/04530566/test"  # predict point shift
    # predict_path = "/disk2/unsupervised reconstruction/samples/bsp_ae_out/04530566/test/128"
    # predict_path = "/disk2/unsupervised reconstruction/samples/bsp_ae_out/04379243/"  # generalization
    # predict_path = "/disk2/unsupervised reconstruction/samples/bsp_ae_out/03001627/test/"  # generalization
    # predict_path = "/disk2/unsupvised scan/samples/bsp_ae_out/"
    # predict_path = "/disk2/unsupervised garment/samples/bsp_ae_out/"
    data_list = []
    for f in sorted(os.listdir(predict_path)):
        if not os.path.isdir(os.path.join(predict_path, f)):
            suffix = f.split(".")[1]
            if suffix == "npy":
                data_list.append(predict_path + "/" + f)
    data_list = data_list[0:1]
    data_list = [predict_path + '/5.npy']

    network = vertex_net(128)
    network.load_state_dict(torch.load("data/checkpoint_aug.pth"), strict=True)
    network.to("cuda")

    for m in network.modules():
        if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
            m.running_mean = None
            m.running_var = None
    network.eval()

    for d in data_list:
        filename = "data/output/" + os.path.split(d)[1][0:-4] + ".off"
        idx = os.path.split(d)[1].split("_")[0]
        # pc = np.loadtxt("/disk2/unsupvised scan/samples/bsp_ae_out/"+idx+"_inputpc.txt", delimiter=";").astype(np.float32)

        data_collection([d], "data/temp.h5", strip=strip)

        vertice_num = 0
        triangles_list = []
        vertices_list = []

        data, label, shift = data_load("data/temp.h5", data_width=data_width)
        batchsize = 2048

        loss_ob = 0
        acc_ob = 0
        shuf_list = torch.randperm(data.shape[0])
        with torch.no_grad():
            for idx in tqdm(range(math.ceil(data.shape[0] / batchsize))):
                data_idx = shuf_list[idx * batchsize:min((idx + 1) * batchsize, data.shape[0])]
                current_size = min((idx + 1) * batchsize, data.shape[0]) - idx * batchsize
                if current_size < 2:
                    continue
                batch_data, batch_label, batch_shift = data[data_idx], label[data_idx], shift[data_idx]
                batch_data = torch.from_numpy(batch_data).to("cuda")
                batch_label = torch.from_numpy(batch_label).to("cuda")
                # batch_data = batch_data.unsqueeze(1)
                # batch_data[:, 0] = batch_data[:, 0] / batch_data[:, 0].reshape(-1, data_width ** 3).mean(-1).unsqueeze(
                #     1).unsqueeze(1).unsqueeze(1)
                # scale = np.random.random() * 1 + 0.5
                # batch_data = batch_data * scale

                out = network(batch_data[:, 0].unsqueeze(1))
                sign, acc = network.predict(out, batch_label)
                # sign = batch_label
                acc_ob += acc.item()
                if construct:
                    sign = sign.detach().cpu().numpy()
                    batch_udf = sign * (batch_data[:, 0, data_width // 2 - 1: data_width // 2 + 2:2,
                                        data_width // 2 - 1:data_width // 2 + 2:2,
                                        data_width // 2 - 1:data_width // 2 + 2:2].reshape(current_size,
                                                                                           -1)).cpu().numpy()
                    # batch_udf = sign * (batch_data[:, 0, data_width // 2: data_width // 2 + 2,
                    #                     data_width // 2:data_width // 2 + 2,
                    #                     data_width // 2:data_width // 2 + 2].reshape(current_size,
                    #                                                                        -1)).cpu().numpy()
                    for iii, pred in enumerate(batch_udf):
                        triangles, vertices = generate_ver_tri(pred)
                        # triangles, vertices = filter_far(triangles, vertices, pc)
                        if len(vertices) > 0:
                            triangles += vertice_num
                            vertice_num += len(vertices)
                            vertices_list.append(vertices)
                            vertices += batch_shift[iii]
                            triangles_list.append(triangles)
        print("acc:%f" % (acc_ob / (idx + 1)))
        if construct:
            triangles_list = np.concatenate(triangles_list, 0)
            vertices_list = np.concatenate(vertices_list, 0)
            # vertices_list -= 1
            vertices_list = vertices_list.astype(np.float32)
            vertices_list /= (grid_dim // strip)
            vertices_list -= 0.5
            # vertices_list, triangles_list = filter_far(vertices_list, triangles_list, pc)
            mesh = trimesh.Trimesh(vertices_list, triangles_list, process=False)
            # normalize
            # bbox = mesh.bounding_box.bounds
            # loc = (bbox[0] + bbox[1]) / 2
            # scale = (bbox[1] - bbox[0]).max()  # / (1 - args.bbox_padding)
            # scale = 1.0 / scale
            # mesh.apply_translation(-loc)
            # mesh.apply_scale(scale)
            ######
            mesh.export(filename)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    data_generation(cls="04530566")
    #
    train()
    # test()
    # construct mesh
    # train_path = "/disk2/unsupervised reconstruction/samples/bsp_ae_out/" \
    #              "04530566_shift_train/8508ec8efeedf4a41ff8f5b5b24b7b46_udf.npy"
    # train_path = "/disk2/unsupervised reconstruction/samples/bsp_ae_out/" \
    #              "04530566_shift_train/973b398bbcc97c3fea9bd4954e1c8c49_udf.npy"
    # pred_path = "/disk2/unsupervised reconstruction/samples/bsp_ae_out/" \
