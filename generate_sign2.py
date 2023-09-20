import open3d as o3d
import numpy as np
import torch
import math
import time
from sklearn.neighbors import KDTree

def cal_min(query_point, triangles_center):
    query_point = torch.tensor(query_point)
    triangles_center = torch.tensor(triangles_center)
    tree = KDTree(triangles_center, leaf_size=16)
    chunk_size = 30000
    G = query_point.shape[0]
    chunk_num = math.ceil(G / chunk_size)
    query_point_chunk = torch.chunk(query_point, chunk_num, 0)
    index_list = []
    dist_list =[]
    for q in query_point_chunk:
        # distance_index = ((q.unsqueeze(1) - triangles_center.unsqueeze(0))**2).sum(-1).min(-1)[1]
        dist, ind = tree.query(q, k=1)
        dist_list.append(dist[:, 0])
        index_list.append(ind[:, 0])
    dist_list = np.concatenate(dist_list, 0)
    index_list = np.concatenate(index_list, 0)
    return dist_list, index_list



def generate_query_point(point_cloud, voxel_dim=256):

    point_cloud = point_cloud + 0.5
    box = np.concatenate([point_cloud.max(axis=0) + 0.05, point_cloud.min(axis=0) - 0.05], 0)
    box = np.clip((box * voxel_dim).round(), a_min=0, a_max=voxel_dim).astype(np.int32)
    grid = np.meshgrid(np.arange(box[3], box[0]), np.arange(box[4], box[1]), np.arange(box[5], box[2]))
    grid = np.stack(grid).reshape(3, -1).transpose(1, 0)
    grid_point = grid.astype(np.float32) / voxel_dim  # + 0.5/float(voxel_dim)
    point_cloud = point_cloud - 0.5
    grid_point = grid_point - 0.5
    return grid_point, grid


def generate_sign(mesh_path, voxel_dim=256):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    bbox = mesh.get_axis_aligned_bounding_box()
    scale = (bbox.max_bound - bbox.min_bound).max()
    mesh.translate(-mesh.get_center())
    mesh.scale(1 / scale*0.9, mesh.get_center())

    mesh.compute_triangle_normals()
    point_cloud = np.array(mesh.sample_points_uniformly(number_of_points=10000).points)
    point_cloud2 = np.array(mesh.sample_points_uniformly(number_of_points=1000000).points)

    query_point, query_index = generate_query_point(point_cloud, voxel_dim)
    vertices, triangles, normal = np.array(mesh.vertices), np.array(mesh.triangles) ,np.array(mesh.triangle_normals)
    # s1, s2 = vertices[1] - vertices[0], vertices[2] - vertices[1]
    # cc = np.cross(s1, s2)
    # cc = cc/np.linalg.norm(cc, ord=2)
    delta = 0.03#015*0.85

    triangle_vertices = vertices[triangles.reshape(-1), :].reshape(-1, 3, 3)
    triangles_center = triangle_vertices.mean(-2).astype(np.float32)
    min_dist, min_index = cal_min(query_point, triangles_center)
    # min_dist = np.linalg.norm(np.expand_dims(query_point, axis=1) - vertices[triangles[min_index].reshape(-1), :].reshape(-1, 3, 3),ord=2, axis=2).min(1)
    chosen_index = np.where(min_dist<delta)
    min_index = min_index[chosen_index]
    min_dist = min_dist[chosen_index]
    query_point = query_point[chosen_index]
    query_index = query_index[chosen_index]

    shift_normal = normal[min_index]
    x0, x1, x2 = triangle_vertices[min_index, 0], triangle_vertices[min_index, 1], triangle_vertices[min_index, 2]
    v1, v2 = (x1-x0)/np.linalg.norm(x1-x0, ord=2, axis=1, keepdims=True), (x2-x0)/np.linalg.norm(x2-x0, ord=2, axis=1, keepdims=True)
    actual_dis = np.matmul(np.expand_dims(x0 - query_point, 1), np.expand_dims(shift_normal, 2))[:, 0]
    min_dist = np.sign(-actual_dis[:, 0]) * min_dist
    p_plane = actual_dis * shift_normal + (
            query_point - x0)
    alpha = np.matmul(np.expand_dims(p_plane, 1), np.expand_dims(v1, 2))[:, 0, 0] / np.linalg.norm(x1-x0, ord=2, axis=1)
    beta = np.matmul(np.expand_dims(p_plane, 1), np.expand_dims(v2, 2))[:, 0, 0] / np.linalg.norm(x2-x0, ord=2, axis=1)
    theta = 5
    condition = (alpha + beta > (-2 * theta)) & (alpha + beta < (1 + theta)) & (alpha > (-2 * theta)) \
                & (alpha  < (1 + theta)) & (beta > (-2 * theta)) & (beta < (1 + theta))
    chosen_index = np.where(condition)

    min_dist = min_dist[chosen_index]
    query_index = query_index[chosen_index]

    return min_dist, query_index, point_cloud, point_cloud2, mesh


if __name__ == '__main__':
    ss = 0
    s = time.time()
    generate_sign("00000000.obj", 256)
    print(time.time()-s)