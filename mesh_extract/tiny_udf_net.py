import math
# import torchvision
import os
from transformer import get_1d_sincos_pos_embed_from_grid
import matplotlib.pyplot as plt
import torch
import argparse
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
from ud_network import vertex_net2, pair_net, transformer_net, vertex_net, vertex_net_even

data_width = 4
grid_dim = -1
data_phase = "256"  # 256

def data_collection(data_list, result_file, strip = 2, train=False):
    data_collected = []
    label_collected = []
    shift_collected = []
    for p in tqdm(data_list):
        data_total = np.load(p)
        if data_total.shape[0] == 4:
            data = data_total[0]
        else:
            data = data_total

        # data = np.expand_dims(data, axis=0)
        dim = data.shape
        global grid_dim
        grid_dim = dim[0]

        ratio = []
        for i_idx in range(0, dim[0]-data_width, strip):
            for j_idx in range(0, dim[1]-data_width, strip):
                for k_idx in range(0, dim[2]-data_width, strip):
                    # if train and np.random.random() > 0.5:
                    #     continue
                    field = data_total[1:, i_idx:i_idx + data_width, j_idx:j_idx + data_width, k_idx:k_idx + data_width]
                    current_grid = np.abs(data_total[0, i_idx:i_idx + data_width, j_idx:j_idx + data_width, k_idx:k_idx + data_width])
                    current_grid = np.concatenate([np.expand_dims(current_grid, axis=0), field], 0)
                    # min_udf = abs_data[i_idx+data_width//2-1:i_idx +data_width//2+2:2,
                    #           j_idx+data_width//2-1:j_idx+data_width//2+2:2, k_idx+data_width//2-1:k_idx +data_width//2+2:2].min()
                    min_udf = current_grid[0].min()  #[1:3,1:3,1:3].min()
                    max_udf = current_grid[0].max()
                    if min_udf > 0.015 or max_udf >= 0.019:
                        continue
                    # grid = np.concatenate([np.expand_dims(current_grid, axis=0), data[1:,i_idx:i_idx + data_width, j_idx:j_idx + data_width, k_idx:k_idx + data_width]], 0)
                    # shift = np.expand_dims(np.array([i_idx, j_idx, k_idx]).astype(np.float32) / 2.0, axis=0)
                    shift = np.expand_dims(np.array([i_idx+1, j_idx+1, k_idx+1]).astype(np.float32) / float(strip), axis=0)
                    label = np.sign(data_total[0, i_idx + data_width // 2 - 1:i_idx + data_width // 2 + 1,
                                    j_idx + data_width // 2 - 1:j_idx + data_width // 2 + 1,
                                    k_idx + data_width // 2 - 1:k_idx + data_width // 2 + 1]).reshape(-1)
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
    data_dir = "samples/bsp_ae_out/udf_data/"+cls+"/"
    # data_dir = "/disk2/unsupvised scan/samples/bsp_ae_out/udf_data_gt/train/"
    # data_dir =  "./samples/bsp_ae_out/"+cls+"/"  #"./data/npy/04530566/"
    # data_dir = "/disk2/occupancy_networks-master/data/ShapeNet.build/04530566/2_binvox/"
    data_list = []
    for x in sorted(os.listdir(data_dir)):
        if x[-3:] == "npy":
            data_list.append(os.path.join(data_dir, x))
    print(data_list)
    return data_list[0:15]


def data_generation(cls):
    train_list = generate_data_list(cls, train=False)
    data_collection(train_list, "data/train_vessel"+data_phase+".h5", train=True)
    # test_list = generate_data_list(cls, train=False)


def data_load(path, data_width=5):
    with h5py.File(path,'r') as f:
        data = f["data"][:].astype(np.float32)
        shift = f["shift"][:].astype(np.float32)
        label = f["label"][:].astype(np.int64)
        # data_width = 5
        # data_shape = data.shape[-1]
        # data = data[:, data_shape//2-data_width//2:data_shape//2+data_width//2+1,
        #        data_shape//2-data_width//2:data_shape//2+data_width//2+1,
        #        data_shape//2-data_width//2:data_shape//2+data_width//2+1]
    return data, label, shift


### density:
### vessel: 0.0204
### all : 0.0229


def train():

    epochs = 30
    network = vertex_net_even(128)
    network.to("cuda")
    # network.load_state_dict(torch.load("data/checkpoint.pth"), strict=True)

    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3,
                                      betas=(0.5, 0.999), weight_decay=0)
    scheduler = CosineAnnealingLR(optimizer, epochs, eta_min=1e-4)
    network.train()
    batchsize = 2048
    data, label, _ = data_load("data/train_vessel"+data_phase+".h5", data_width=data_width)

    for epoch in range(epochs):
        scheduler.step()
        loss_ob = 0
        acc_ob = 0
        shuf_list = torch.randperm(data.shape[0])
        iter_num = min(data.shape[0]//batchsize, 2000)
        for idx in tqdm(range(iter_num)):

            data_idx = shuf_list[idx*batchsize:(idx+1)*batchsize]
            batch_data, batch_label = data[data_idx], label[data_idx]
            batch_data = torch.from_numpy(batch_data).to("cuda")
            batch_label = torch.from_numpy(batch_label).to("cuda")
            batch_data = batch_data + (torch.rand_like(batch_data)-0.5)*0.0000
            batch_data[:, 0] = F.relu(batch_data[:, 0])
            ############ augmentation ##############
            flip_dim = int(np.random.random()*4)%4
            batch_label = batch_label.reshape(batchsize, 2, 2, 2)
            if flip_dim<3:
                batch_data = torch.flip(batch_data, (flip_dim + 2,))
                batch_data[:, flip_dim + 1] = -batch_data[:, flip_dim + 1]
                batch_label = torch.flip(batch_label, (flip_dim + 1,))
            #
            # random rotation
            rot_dim = int(np.random.random() * 4) % 4
            if rot_dim < 3:
                batch_data = torch.rot90(batch_data, 1, dims=[rot_dim+2, (rot_dim+1)%3+2])
                batch_label = torch.rot90(batch_label, 1, dims=[rot_dim+1, (rot_dim+1)%3+1])
                if rot_dim == 0: # z
                    matrix = torch.tensor([[ 0, -1, 0 ],
                                       [ 1, 0, 0 ],
                                       [ 0, 0, 1 ]]).to(batch_data.device).float()
                if rot_dim == 1:  # x
                    matrix = torch.tensor([[ 1, 0, 0 ],
                                       [ 0, 0, -1 ],
                                       [ 0, 1, 0 ]]).to(batch_data.device).float()
                if rot_dim == 2:  # y
                    matrix = torch.tensor([[ 0, 0, 1 ],
                                       [ 0, 1, 0 ],
                                       [ -1, 0, 0 ]]).to(batch_data.device).float()
                B, C, D, _, _ = batch_data.shape
                rot_shift = torch.matmul(matrix.unsqueeze(0).repeat(B, 1, 1), batch_data[:,1:].reshape(B, 3, D**3)).reshape(B, 3, D,D,D)
                batch_data[:,1:] = rot_shift

            batch_label = batch_label.reshape(batchsize, -1)
            ########################################
            out = network(batch_data)
            loss= network.cal_loss(out, batch_label)
            network.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ob += loss.item()
            if epoch % 10 == 9:
                torch.save(network.state_dict(), "data/checkpoint"+data_phase+".pth")
        print("Epoch:%d, loss:%f"%(epoch, loss_ob/(idx+1)))
        torch.save(network.state_dict(), "data/checkpoint"+data_phase+".pth")


def test(predict_path = "./data/npy/", cls="001"):
    if data_width %2 == 1:
        strip = 2
    else:
        strip = 1
    construct = True
    predict_path += cls #+"_gt"
    # predict_path = "/disk2/unsupervised reconstruction/samples/bsp_ae_out/udf_data/train/"
    # predict_path = "/disk2/unsupervised reconstruction/samples/bsp_ae_out/udf_data/test/"

    # predict_path = "/disk2/unsupervised reconstruction/samples/bsp_ae_out/04530566/train/"  # calculate point shift
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
                data_list.append(predict_path+"/"+f)
    # data_list = [data_list[2]]

    network = vertex_net_even(128)
    network.load_state_dict(torch.load("data/checkpoint"+data_phase+".pth"), strict=True)
    network.to("cuda")

    # for m in network.modules():
    #     if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
    #         m.running_mean = None
    #         m.running_var = None
    network.eval()

    for d in data_list:
        if not os.path.exists("data/output/"+cls):
            os.mkdir("data/output/"+cls)
        uuid = os.path.split(d)[1][0:-4]
        data_collection([d], "data/temp.h5", strip=strip)
        filename = predict_path + "/" + uuid + ".off"

        vertice_num = 0
        triangles_list = []
        vertices_list = []

        data, label, shift = data_load("data/temp.h5", data_width=data_width)
        batchsize = 2048

        loss_ob = 0
        acc_ob = 0
        # shuf_list = torch.randperm(data.shape[0])
        shuf_list = torch.arange(data.shape[0])
        with torch.no_grad():
            for idx in tqdm(range(math.ceil(data.shape[0] / batchsize))):
                data_idx = shuf_list[idx * batchsize:min((idx + 1) * batchsize, data.shape[0])]
                current_size = min((idx + 1) * batchsize, data.shape[0]) - idx * batchsize
                if current_size < 2:
                    continue
                batch_data, batch_label, batch_shift = data[data_idx], label[data_idx], shift[data_idx]
                batch_data = torch.from_numpy(batch_data).to("cuda")
                batch_label = torch.from_numpy(batch_label).to("cuda")

                out = network(batch_data)
                sign = network.predict(out, batch_label)

                global grid_dim
                grid_dim = 256

                if construct:
                    past_grid = -1*np.ones((3, grid_dim,grid_dim,grid_dim)).astype(np.int64)
                    sign = sign.detach().cpu().numpy()
                    batch_udf = sign * (batch_data[:, 0, data_width // 2 - 1: data_width // 2 + 1,
                                         data_width // 2 - 1:data_width // 2 + 1,
                                         data_width // 2 - 1:data_width // 2 + 1].reshape(current_size, -1)).cpu().numpy()
                    # batch_udf = sign * (batch_data[:, 0, data_width // 2: data_width // 2 + 2,
                    #                     data_width // 2:data_width // 2 + 2,
                    #                     data_width // 2:data_width // 2 + 2].reshape(current_size,
                    #                                                                        -1)).cpu().numpy()
                    # np.save("tttt.npy", batch_udf.detach().cpu().numpy())
                    # batch_udf = np.load("tttt.npy")
                    for iii, pred in enumerate(batch_udf):
                        if np.sign(pred).sum() > 7.5 or np.sign(pred).sum() < -7.5:
                            continue
                        else:
                            left_up_index = batch_shift[iii, 0]
                            triangles, vertices, past_grid, vertice_num = generate_ver_tri(pred, left_up_index, past_grid, vertice_num)
                            if vertices.size > 0:
                                vertices_list.append(vertices)
                            triangles_list.append(triangles)
        if construct:
            # ss = vertices_list[1087]
            triangles_list = np.concatenate(triangles_list, 0)
            vertices_list = np.concatenate(vertices_list, 0)
            # np.save("triangles_list.npy", triangles_list)
            vertices_list = vertices_list.astype(np.float32)
            # vertices_list += 0.5
            vertices_list /= (grid_dim//strip)
            vertices_list -= 0.5
            mesh = trimesh.Trimesh(vertices_list, triangles_list, process=False)
            mesh.remove_degenerate_faces()
            # normalize
            # bbox = mesh.bounding_box.bounds
            # loc = (bbox[0] + bbox[1]) / 2
            # scale = (bbox[1] - bbox[0]).max()  # / (1 - args.bbox_padding)
            # scale = 1.0 / scale
            # mesh.apply_translation(-loc)
            # mesh.apply_scale(scale)
            ######
            # loc_path = "/disk2/occupancy_networks-master/data/ShapeNet/"
            # pc = np.load(os.path.join(loc_path + cls + "/" + uuid.split("_")[0] + "/", "pointcloud.npz"), allow_pickle=False)
            # loc, scale = pc["loc"], pc["scale"]
            # mesh.apply_translation(-loc)
            # mesh.apply_scale(1.0 / scale)
            mesh.export(filename)


def generate_shapenet(path="/disk2/unsupervised reconstruction/samples/bsp_ae_out/"):
    # cls_list = ["03636649", "03691459",
    #                        "04090263", "04256520", "04379243", "04401088", "04530566"]  #
    cls_list = ["02691156", "02828884", "02933112","02958343", "03001627", "03211117"]
    # cls_list = ["04379243"]
    for cls in cls_list:
        test(path, cls)


def generate_example():
    f=open("../CAP-UDF-master/shapenet_example.txt", "r")
    examples = f.readlines()
    root_path = "/disk2/unsuper_3fold/samples/bsp_ae_out/"
    data_list = [root_path + x[0:-1]+"_udf.npy" for x in examples]
    if data_width %2 == 1:
        strip = 2
    else:
        strip = 1
    construct = True
    network = vertex_net_even(128)
    network.load_state_dict(torch.load("data/checkpoint"+data_phase+".pth"), strict=True)
    network.to("cuda")

    # for m in network.modules():
    #     if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
    #         m.running_mean = None
    #         m.running_var = None
    network.eval()

    for d in tqdm(data_list):
        cls = os.path.split(os.path.split(d)[0])[1]
        if not os.path.exists("data/vis_example/"+cls):
            os.makedirs("data/vis_example/"+cls)
        uuid = os.path.split(d)[1][0:-4]
        data_collection([d], "data/temp_vis.h5", strip=strip)
        filename = "data/vis_example/"+ cls+"/" + uuid + "_udf.off"

        vertice_num = 0
        triangles_list = []
        vertices_list = []

        data, label, shift = data_load("data/temp_vis.h5", data_width=data_width)
        batchsize = 2048

        loss_ob = 0
        acc_ob = 0
        # shuf_list = torch.randperm(data.shape[0])
        shuf_list = torch.arange(data.shape[0])
        with torch.no_grad():
            for idx in tqdm(range(math.ceil(data.shape[0] / batchsize))):
                data_idx = shuf_list[idx * batchsize:min((idx + 1) * batchsize, data.shape[0])]
                current_size = min((idx + 1) * batchsize, data.shape[0]) - idx * batchsize
                if current_size < 2:
                    continue
                batch_data, batch_label, batch_shift = data[data_idx], label[data_idx], shift[data_idx]
                batch_data = torch.from_numpy(batch_data).to("cuda")
                batch_label = torch.from_numpy(batch_label).to("cuda")
                out = network(batch_data)
                sign, acc = network.predict(out, batch_label)
                # sign = batch_label

                global grid_dim
                grid_dim = 256

                acc_ob += acc.item()
                if construct:
                    past_grid = -1*np.ones((3, grid_dim,grid_dim,grid_dim)).astype(np.int64)
                    sign = sign.detach().cpu().numpy()
                    batch_udf = sign * (batch_data[:, 0, data_width // 2 - 1: data_width // 2 + 1,
                                         data_width // 2 - 1:data_width // 2 + 1,
                                         data_width // 2 - 1:data_width // 2 + 1].reshape(current_size, -1)).cpu().numpy()
                    for iii, pred in enumerate(batch_udf):
                        if np.sign(pred).sum() > 7.5 or np.sign(pred).sum() < -7.5:
                            continue
                        else:
                            left_up_index = batch_shift[iii, 0]
                            triangles, vertices, past_grid, vertice_num = generate_ver_tri(pred, left_up_index, past_grid, vertice_num)
                            if vertices.size > 0:
                                vertices_list.append(vertices)
                            triangles_list.append(triangles)
        print("acc:%f" % (acc_ob / (idx+1)))
        if construct:
            # ss = vertices_list[1087]
            triangles_list = np.concatenate(triangles_list, 0)
            vertices_list = np.concatenate(vertices_list, 0)
            # np.save("triangles_list.npy", triangles_list)
            vertices_list = vertices_list.astype(np.float32)
            # vertices_list += 0.5
            vertices_list /= (grid_dim//strip)
            vertices_list -= 0.5
            mesh = trimesh.Trimesh(vertices_list, triangles_list, process=False)
            mesh.remove_degenerate_faces()
            # normalize
            # bbox = mesh.bounding_box.bounds
            # loc = (bbox[0] + bbox[1]) / 2
            # scale = (bbox[1] - bbox[0]).max()  # / (1 - args.bbox_padding)
            # scale = 1.0 / scale
            # mesh.apply_translation(-loc)
            # mesh.apply_scale(scale)
            ######
            # loc_path = "/disk2/occupancy_networks-master/data/ShapeNet/"
            # pc = np.load(os.path.join(loc_path + cls + "/" + uuid.split("_")[0] + "/", "pointcloud.npz"), allow_pickle=False)
            # loc, scale = pc["loc"], pc["scale"]
            # mesh.apply_translation(-loc)
            # mesh.apply_scale(1.0 / scale)
            mesh.export(filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", dest="train", default=False,
                        help="True for training, False for testing [False]")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    if args.train:
        data_generation(cls="test")
        train()
    else:
        test("samples/bsp_ae_out/udf_data/", cls="test")
