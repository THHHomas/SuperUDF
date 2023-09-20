import mcubes
import os
import h5py
import open3d as o3d
import trimesh
import numpy as np
from tqdm import tqdm
from generate_sign2 import generate_sign
import matplotlib.pyplot as plt


def sample_point(mesh):
    sample_num = 10000
    point_cloud, face_idx = trimesh.sample.sample_surface_even(mesh, sample_num+500)
    N = point_cloud.shape[0]
    if N < sample_num:
        choice = np.random.choice(N, sample_num - N)
        point_cloud = np.concatenate([point_cloud, point_cloud[choice, :]], 0)
        face_idx = np.concatenate([face_idx, face_idx[choice]], 0)
    else:
        point_cloud = point_cloud[0:sample_num]
        face_idx = face_idx[0:sample_num]
    normal = mesh.face_normals[face_idx]
    point_cloud = np.concatenate([point_cloud, normal], -1)
    return point_cloud


def sample_uniform_point(data_path_list, phase="train"):
    point_cloud_list = []
    point_cloud_list2 = []
    mesh_path_list = []
    for data_path in data_path_list:
        for object_id in tqdm(sorted(os.listdir(data_path))):
            mesh_path = os.path.join(data_path, object_id, object_id+"_vh_clean_2.ply")
            mesh = trimesh.load(mesh_path, process=False)
            # mesh.apply_translation(-mesh.centroid)
            bbox = mesh.bounding_box.bounds
            loc = (bbox[0] + bbox[1]) / 2
            scale = (bbox[1] - bbox[0]).max()
            # Transform input mesh
            mesh.apply_translation(-loc)
            mesh.apply_scale(1 / scale)

            point_cloud = sample_point(mesh)
            point_cloud2 = sample_point(mesh)
            point_cloud_list.append(point_cloud)
            point_cloud_list2.append(point_cloud2)
            mesh_path_list.append(mesh_path)
            # np.savetxt("data/sample.txt", point_cloud, delimiter=";")
            # exit(0)
    point_cloud_list = np.stack(point_cloud_list)
    point_cloud_list2 = np.stack(point_cloud_list2)

    total_num = point_cloud_list.shape[0]
    with open("data/mesh_path_"+phase+".txt", "w") as f:
        f.writelines(mesh_path_list[0:total_num])

    with h5py.File("data/"+phase+".h5", "w") as f:
        f.create_dataset("point_cloud", data=point_cloud_list[0:total_num])
        f.create_dataset("point_cloud2", data=point_cloud_list2[0:total_num])


def sample_point_with_ss(data_path_list, phase="train", voxel_dim = 256):
    gt_path = '/disk2/unsupvised scan/data/gt/'+phase+'/'
    point_cloud_list = []
    point_cloud_list2 = []
    mesh_path_list = []
    sign_list = []
    mesh_mun =0
    # ss =  sorted(os.listdir(data_path_list[0]))[5:]
    for data_path in data_path_list:
        for object_id in tqdm(sorted(os.listdir(data_path))):
            mesh_path = os.path.join(data_path, object_id, object_id+"_vh_clean_2.ply")
            result = np.ones((voxel_dim, voxel_dim, voxel_dim))*0.03
            min_dist, query_index, point_cloud, point_cloud2, mesh = generate_sign(mesh_path, voxel_dim)
            result[query_index[:, 0], query_index[:, 1], query_index[:, 2]] = min_dist
            result = result.astype(np.float32)
            # clamp = result[voxel_dim // 2]
            # plt.imshow(np.abs(clamp), cmap=plt.cm.jet)
            # plt.savefig("x" + str(0) + "_gt.png")
            # exit(0)
            np.save("data/npy/"+str(mesh_mun) +".npy", result)
            # if mesh_mun == 1:
            #     o3d.io.write_triangle_mesh("0.off", mesh)
            #     np.savetxt("0.txt", point_cloud, delimiter=";")
            #     exit(0)
            o3d.io.write_triangle_mesh(gt_path+str(mesh_mun)+"_gt.off", mesh)
            mesh_mun += 1
            # vertices, triangles = mcubes.marching_cubes(result, 0)
            # vertices /= np.array([voxel_dim, voxel_dim, voxel_dim])
            # vertices -= 0.5
            # # # Undo padding
            # # matrix = rotation(-math.pi / 2).numpy()
            # # vertices = np.matmul(vertices, matrix)
            # mesh = trimesh.Trimesh(vertices, triangles, process=False)
            # mesh.export("0.off")
            # exit(0)

            point_cloud_list.append(point_cloud)
            point_cloud_list2.append(point_cloud2)
            sign_list.append(result)
            mesh_path_list.append(mesh_path+"\n")
            # np.savetxt("data/sample.txt", point_cloud, delimiter=";")
    # exit(0)
    point_cloud_list = np.stack(point_cloud_list)
    point_cloud_list2 = np.stack(point_cloud_list2)
    sign_list = np.stack(sign_list)

    total_num = len(mesh_path_list)
    with open("data/mesh_path_"+phase+".txt", "w") as f:
        f.writelines(mesh_path_list[0:total_num])

    with h5py.File("data/"+phase+".h5", "w") as f:
        f.create_dataset("point_cloud", data=point_cloud_list[0:total_num])
        f.create_dataset("point_cloud2", data=point_cloud_list2[0:total_num])
        f.create_dataset("sign", data=sign_list[0:total_num])


if __name__ == '__main__':
    # sample_uniform_point(["../scans_test/scans_test", "../scans_test/scans_test_1"], phase="train")
    # sample_uniform_point(["../scans_test/scans_test_2"], phase="test")

    # sample_point_with_ss(["../scans_test/scans_test" ], phase="train", voxel_dim=4)
    sample_point_with_ss(["../scans_test/scans_test_1", "../scans_test/scans_test_2"], phase="test", voxel_dim=256)
