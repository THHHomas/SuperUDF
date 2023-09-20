import torch
import numpy as np
import os
import trimesh
# from sklearn.neighbors import KDTree
from libkdtree import KDTree
import open3d as o3d
from utils import sample_points, read_ply_polygon, write_ply_polygon
# from mesh_intersection import check_mesh_contains, compute_iou


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                    for g in scene_or_mesh.geometry.values()))
    else:
        assert(isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = scene_or_mesh
    return mesh


def l2_distance(src, dst):
    """
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist.sqrt()


def l1_distance(src, dst):
    """
    Calculate l1 distance between each two points.

        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    dist = (src.unsqueeze(2) - dst.unsqueeze(1)).abs().sum(-1)
    return dist


def charmer_distance(pc1, pc2):
    cloeset_dis = l1_distance(pc1[:,:,0:3], pc2[:,:,0:3])
    ver_closest_dis, ver_closest_idx = cloeset_dis.min(dim=-2)
    cloeset_dis, cloeset_idx = cloeset_dis.min(dim=-1)

    normal1 = pc1[:, :,3:]
    normal2 = pc2[:, cloeset_idx.squeeze(), 3:]
    nc = torch.matmul(normal1.unsqueeze(2), normal2.unsqueeze(3)).squeeze().unsqueeze(0)

    normal1 = pc1[:, ver_closest_idx.squeeze(), 3:]
    normal2 = pc2[:, :, 3:]
    nc_ver = torch.matmul(normal1.unsqueeze(2), normal2.unsqueeze(3)).squeeze().unsqueeze(0)
    ss = nc_ver.mean()
    ss2 = nc.mean()
    nc = (ss + ss2) / 2
    cd = (cloeset_dis.mean() + ver_closest_dis.mean()) / 2
    return cd, nc


def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)
    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, normals_dot_product


def evaluate_cd_nc_garment(path, original_mehs_path, topk = 3, chosen_suffix=".off", log_file="abc_normat"):
    cd_list = []
    nc_list = []

    idx_list = []
    for ff in os.listdir(path):
        idx, suffix = ff.split(".")
        if suffix == "off":
            idx, _ = idx.split("_")
            idx_list.append(idx)
    # test_file_path = "/disk2/unsupervised reconstruction/test.lst"
    # with open(test_file_path, "r") as f:
    #     idx_list = f.readlines()
    # idx_list = [x[0:-1] for x in idx_list if len(x)>2]

    sample_num = 0
    test_list = sorted(idx_list)
    # print(len(test_list))
    # exit(0)
    for idx in test_list:
        idx = str(idx)
        print(idx)
        uuid = idx
        mesh = trimesh.load(path + "/" + str(uuid) + chosen_suffix)
        bbox = mesh.bounding_box.bounds
        loc = (bbox[0] + bbox[1]) / 2
        scale = (bbox[1] - bbox[0]).max()  # / (1 - args.bbox_padding)
        scale = 1.0 / scale
        mesh.apply_translation(-loc)
        mesh.apply_scale(scale)
        points, face_idx = mesh.sample(100000, return_index=True)
        normals = mesh.face_normals[face_idx]
        # mesh.export(uuid+"udf"+".off")
        # # mesh = mesh.filter_smooth_laplacian(number_of_iterations=1)
        # vertices, polygons = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
        # # vertices, polygons = read_ply_polygon(path + "/" + str(idx) + "_vox.ply")
        # pc_vox = sample_points(vertices, polygons, 100000)

        pc1=[np.concatenate([points, normals], -1).astype(np.float32)]

        mesh = trimesh.load(original_mehs_path + str(uuid) + "_gt.off")
        # bbox = mesh.bounding_box.bounds
        # loc = (bbox[0] + bbox[1]) / 2
        # scale = (bbox[1] - bbox[0]).max()  # / (1 - args.bbox_padding)
        # scale = 1.0 / scale
        mesh.apply_translation(-loc)
        mesh.apply_scale(scale)
        # mesh.export(uuid+".off")
        # mesh = mesh.filter_smooth_laplacian(number_of_iterations=3)
        points, face_idx = mesh.sample(100000, return_index=True)
        normals = mesh.face_normals[face_idx]
        pc2 = [np.concatenate([points, normals], -1).astype(np.float32)]

        pc1 = torch.tensor(pc1)
        pc2 = torch.tensor(pc2)
        pc1 = pc1.numpy()
        pc2 = pc2.numpy()
        # np.savetxt("recon.txt", pc1[0], delimiter=";")
        # np.savetxt("gt.txt", pc2[0], delimiter=";")
        # exit(0)

        pointcloud_tgt = pc2[0,:, 0:3]
        normals_tgt = pc2[0,:,3:]
        pointcloud = pc1[0,:,0:3]
        normals = pc1[0,:,3:]
        completeness, completeness_normals = distance_p2p(
            pointcloud_tgt, normals_tgt, pointcloud, normals
        )
        accuracy, accuracy_normals = distance_p2p(
            pointcloud, normals, pointcloud_tgt, normals_tgt
        )
        completeness = completeness.mean()
        completeness_normals = np.abs(completeness_normals).mean()
        accuracy = accuracy.mean()
        accuracy_normals = np.abs(accuracy_normals).mean()
        chamferL1 = 0.5 * (completeness + accuracy)
        normals_correctness = (
                0.5 * completeness_normals + 0.5 * accuracy_normals
        )

        # chamferL1, normals_correctness = charmer_distance(pc1, pc2)
        nc_list.append(normals_correctness)
        cd_list.append(chamferL1)
        sample_num += 1

        print("finished:  %.2f%%"%(100*float(sample_num)/len(test_list)))

    nc_list = np.array(nc_list)
    cd_list = np.array(cd_list)

    print(nc_list)
    print(cd_list)
    nc_total = nc_list.mean()
    cd_total = cd_list.mean()
    with open("data/"+log_file+".log", "w") as f:
        f.write("cd is %f:"%cd_total.item()+" nc is: %f\n"%nc_total.item())
        f.write("cd top k:\n")

    return cd_total, nc_total


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # test_file = "data/data_per_category/data_per_category/02691156_airplane/02691156_vox256_img_test.txt"
    # test_file = "data/data_per_category/data_per_category/00000000_all/00000000_vox256_img_test.txt"
    # cd, nc, iou = evaluate_cd_nc("samples/bsp_ae_out", test_file)

    cd, nc = evaluate_cd_nc_garment("./data/output/ling_64/udf_data/test/",
                                     "./data/gt/test/", chosen_suffix="_udf.off",
                                     log_file="scan_udf")
    # cd, nc = evaluate_cd_nc_garment("/disk2/unsupvised scan/data/PSR/",
    #                                 "/disk2/unsupvised scan/data/gt/test/", chosen_suffix="_psr.off",
    #                                 log_file="scan_psr")
    # cd, nc = evaluate_cd_nc_garment("/disk2/unsupvised scan/data/BPA/",
    #                                 "/disk2/unsupvised scan/data/gt/test/", chosen_suffix="_bpa.off",
    #                                 log_file="scan_bpa")
    # cd, nc = evaluate_cd_nc_garment("/disk2/unsupvised scan/data/NDF/",
    #                                 "/disk2/unsupvised scan/data/gt/test/", chosen_suffix="_bpa.off",
    #                                 log_file="scan_ndf")
    # cd, nc = evaluate_cd_nc_garment("/disk2/unsupvised scan/data/BPA_1000/",
    #                                 "/disk2/unsupvised scan/data/gt/test/", chosen_suffix="_bpa.off",
    #                                 log_file="scan_bpa1000")
    # cd, nc = evaluate_cd_nc_garment("/disk2/unsupvised scan/data/PSR_1000/",
    #                                 "/disk2/unsupvised scan/data/gt/test/", chosen_suffix="_psr.off",
    #                                 log_file="scan_psr1000")
    # cd, nc = evaluate_cd_nc_garment("/disk2/CAP-UDF-master/scannet_vis/",
    #                                 "data/gt/train/", chosen_suffix="_cap.off",
    #                                 log_file="scannet_cap")
    # cd, nc = evaluate_cd_nc_garment("/disk2/gifs-main/scannet_vis/",
    #                                 "data/gt/train/", chosen_suffix="_gifs.off",
    #                                 log_file="scannet_gifs")
    print("cd is:", cd.item(), " nc is:", nc.item())