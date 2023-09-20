import mcubes
import numpy as np
import math
import os
import torch
import trimesh
import h5py

import torch.nn.functional as F
from torch import nn
from libkdtree import KDTree


# from generate_sign2 import cal_udf


def octree(point, x_boundary, y_boundary, z_boundary, size, octant_list=[], octant_info=[]):
    if point.shape[0] >= 50:
        x_index = point[:, 0] > x_boundary
        y_index = point[:, 1] > y_boundary
        z_index = point[:, 2] > z_boundary
        octant0 = point[torch.where(x_index & y_index & z_index)]
        octant1 = point[torch.where(x_index & ~y_index & z_index)]
        octant2 = point[torch.where(x_index & y_index & ~z_index)]
        octant3 = point[torch.where(x_index & ~y_index & ~z_index)]
        octant4 = point[torch.where(~x_index & ~y_index & z_index)]
        octant5 = point[torch.where(~x_index & y_index & ~z_index)]
        octant6 = point[torch.where(~x_index & ~y_index & ~z_index)]
        octant7 = point[torch.where(~x_index & y_index & z_index)]

        # octant0 = point[np.where((point[:,0] > x_boundary-size/4) & (point[:,1] > y_boundary-size/4) & (point[:,2] > z_boundary-size/4))]
        # octant1 = point[np.where((point[:,0] > x_boundary-size/4) & (point[:,1] < y_boundary+size/4) & (point[:,2] > z_boundary-size/4))]
        # octant2 = point[np.where((point[:,0] > x_boundary-size/4) & (point[:,1] > y_boundary-size/4) & (point[:,2] < z_boundary+size/4))]
        # octant3 = point[np.where((point[:,0] > x_boundary-size/4) & (point[:,1] < y_boundary+size/4) & (point[:,2] < z_boundary+size/4))]
        # octant4 = point[np.where((point[:,0] < x_boundary+size/4) & (point[:,1] < y_boundary+size/4) & (point[:,2] > z_boundary-size/4))]
        # octant5 = point[np.where((point[:,0] < x_boundary+size/4) & (point[:,1] > y_boundary-size/4) & (point[:,2] < z_boundary+size/4))]
        # octant6 = point[np.where((point[:,0] < x_boundary+size/4) & (point[:,1] < y_boundary+size/4) & (point[:,2] < z_boundary+size/4))]
        # octant7 = point[np.where((point[:,0] < x_boundary+size/4) & (point[:,1] > y_boundary-size/4) & (point[:,2] > z_boundary-size/4))]

        percent = 1 / 4
        octree(octant0, x_boundary + size * percent, y_boundary + size * percent, z_boundary + size * percent, size / 2,
               octant_list, octant_info)
        octree(octant1, x_boundary + size * percent, y_boundary - size * percent, z_boundary + size * percent, size / 2,
               octant_list, octant_info)
        octree(octant2, x_boundary + size * percent, y_boundary + size * percent, z_boundary - size * percent, size / 2,
               octant_list, octant_info)
        octree(octant3, x_boundary + size * percent, y_boundary - size * percent, z_boundary - size * percent, size / 2,
               octant_list, octant_info)
        octree(octant4, x_boundary - size * percent, y_boundary - size * percent, z_boundary + size * percent, size / 2,
               octant_list, octant_info)
        octree(octant5, x_boundary - size * percent, y_boundary + size * percent, z_boundary - size * percent, size / 2,
               octant_list, octant_info)
        octree(octant6, x_boundary - size * percent, y_boundary - size * percent, z_boundary - size * percent, size / 2,
               octant_list, octant_info)
        octree(octant7, x_boundary - size * percent, y_boundary + size * percent, z_boundary + size * percent, size / 2,
               octant_list, octant_info)

    elif 50 > point.shape[0] > 0:
        octant_list.append(point)
        octant_info.append(np.array([x_boundary, y_boundary, z_boundary, size]))

    return octant_list, octant_info


def get_extended_point(point, octant_info, extend_size=1):
    point_list = []
    for octant in octant_info:
        center = octant[0:3]
        size = octant[3] / 2 * extend_size
        left_bottom = center - size
        right_up = center + size
        index = (point[:, 0] > left_bottom[0]) & (point[:, 1] > left_bottom[1])&(point[:, 2] > left_bottom[2]) &\
                (right_up[0] >= point[:, 0]) & (right_up[1] >= point[:, 1]) & (right_up[2] >= point[:, 2])
        index = torch.where(index)
        chosen_point = point[index]
        point_list.append(chosen_point)
    return point_list


def get_extended_point_with_label(point, octant_info, extend_size=1):
    point_list = []
    label_list = []
    for octant in octant_info:
        center = octant[0:3]
        size = octant[3] / 2 * extend_size
        left_bottom = center - size
        right_up = center + size
        index = (point[:, 0] > left_bottom[0]) & (point[:, 1] > left_bottom[1])&(point[:, 2] > left_bottom[2]) &\
                (right_up[0] >= point[:, 0]) & (right_up[1] >= point[:, 1]) & (right_up[2] >= point[:, 2])
        index = torch.where(index)
        temp = point[index]
        chosen_point = temp[:, 0:3]
        chosen_label = temp[:, 3]
        point_list.append(chosen_point)
        label_list.append(chosen_label)
    return point_list, label_list


def rand_rotation():
    matrix = []
    theta = np.random.random()*2*math.pi
    matrix.append(np.array([math.cos(theta), 0, math.sin(theta)]))
    matrix.append(np.array([0, 1, 0]))
    matrix.append(np.array([math.sin(-theta), 0, math.cos(theta)]))
    matrix = torch.from_numpy(np.array(matrix).T).float()
    res = matrix

    matrix = []
    theta = np.random.random() * 2 * math.pi
    matrix.append(np.array([1, 0, 0]))
    matrix.append(np.array([0, math.cos(theta), math.sin(-theta)]))
    matrix.append(np.array([0, math.sin(theta), math.cos(theta)]))
    matrix = torch.from_numpy(np.array(matrix).T).float()
    res = torch.matmul(res, matrix)

    matrix = []
    theta = np.random.random() * 2 * math.pi
    matrix.append(np.array([math.cos(theta), math.sin(-theta), 0]))
    matrix.append(np.array([math.sin(theta), math.cos(theta), 0]))
    matrix.append(np.array([0, 0, 1]))
    matrix = torch.from_numpy(np.array(matrix).T).float()
    res = torch.matmul(res, matrix)
    return res


def interpolation(xyz, new_xyz, feat, k=6):
    # assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()
    # idx, dist = knnquery(k, xyz, new_xyz, offset, new_offset) # (n, 3), (n, 3)
    idx, dist = knn_point(k, xyz, new_xyz, dis=True)
    dist_recip = 1.0 / (dist + 1e-8) # (n, 3)
    norm = torch.sum(dist_recip, dim=1, keepdim=True)
    weight = dist_recip / norm # (n, 3)
    new_feat = index_points(feat, idx)
    new_feat = torch.matmul(weight.unsqueeze(2), new_feat)[:,:,0,:]
    return new_feat


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def read_voxel(name, dim, binvox_path="/disk2/occupancy_networks-master/data/ShapeNet/"):
    binvox_file = os.path.join(binvox_path, name, "model.binvox")
    voxel = binvox_rw.read_as_3d_array(binvox_file).data.astype(np.float32)
    stride = 256//dim
    w = torch.ones(1, 1, stride, stride, stride).float()
    conv = F.conv3d(torch.from_numpy(voxel).unsqueeze(0).unsqueeze(0), w, stride=stride).ceil()
    voxel = torch.clamp(conv, max=1).squeeze().numpy()
    return voxel


def square_distance(src, dst):
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


def knn_point(nsample, xyz, new_xyz, dis=False, sorted=True):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    with torch.no_grad():
        sqrdists = square_distance(new_xyz, xyz)
        g_dis, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=sorted)
        if dis:
            return group_idx, g_dis
        return group_idx


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


# .ply format  --  X,Y,Z, normalX,normalY,normalZ
def parse_ply_planes(shape_name, num_of_points=2048):
    file = open(shape_name, 'r')
    lines = file.readlines()
    vertices = np.zeros([num_of_points, 7], np.float32)
    assert lines[9].strip() == "end_header"
    for i in range(num_of_points):
        line = lines[i + 10].split()
        vertices[i, 0] = float(line[0])  # X
        vertices[i, 1] = float(line[1])  # Y
        vertices[i, 2] = float(line[2])  # Z
        vertices[i, 3] = float(line[3])  # normalX
        vertices[i, 4] = float(line[4])  # normalY
        vertices[i, 5] = float(line[5])  # normalZ
        tmp = vertices[i, 0] * vertices[i, 3] + vertices[i, 1] * vertices[i, 4] + vertices[i, 2] * vertices[i, 5]
        vertices[i, 6] = -tmp  # d for plane ax+by+cz+d = 0
    return vertices


def parse_ply_list_to_planes(ref_txt_name, data_dir, data_txt_name):
    # open file & read points
    ref_file = open(ref_txt_name, 'r')
    ref_names = [line.strip() for line in ref_file]
    ref_file.close()
    data_file = open(data_txt_name, 'r')
    data_names = [line.strip() for line in data_file]
    data_file.close()

    num_shapes = len(ref_names)
    ref_points = np.zeros([num_shapes, 2048, 7], np.float32)
    idx = np.zeros([num_shapes], np.int32)

    for i in range(num_shapes):
        shape_name = data_dir + "/" + ref_names[i] + ".ply"
        shape_idx = data_names.index(ref_names[i])
        shape_planes = parse_ply_planes(shape_name)
        ref_points[i, :, :] = shape_planes
        idx[i] = shape_idx

    return ref_points, idx, ref_names


def write_ply_point(name, vertices):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex " + str(len(vertices)) + "\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(str(vertices[ii, 0]) + " " + str(vertices[ii, 1]) + " " + str(vertices[ii, 2]) + "\n")
    fout.close()


def write_ply_point_normal(name, vertices, normals=None):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex " + str(len(vertices)) + "\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("property float nx\n")
    fout.write("property float ny\n")
    fout.write("property float nz\n")
    fout.write("end_header\n")
    if normals is None:
        for ii in range(len(vertices)):
            fout.write(str(vertices[ii, 0]) + " " + str(vertices[ii, 1]) + " " + str(vertices[ii, 2]) + " " + str(
                vertices[ii, 3]) + " " + str(vertices[ii, 4]) + " " + str(vertices[ii, 5]) + "\n")
    else:
        for ii in range(len(vertices)):
            fout.write(str(vertices[ii, 0]) + " " + str(vertices[ii, 1]) + " " + str(vertices[ii, 2]) + " " + str(
                normals[ii, 0]) + " " + str(normals[ii, 1]) + " " + str(normals[ii, 2]) + "\n")
    fout.close()


def write_ply_triangle(name, vertices, triangles):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex " + str(len(vertices)) + "\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("element face " + str(len(triangles)) + "\n")
    fout.write("property list uchar int vertex_index\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(str(vertices[ii, 0]) + " " + str(vertices[ii, 1]) + " " + str(vertices[ii, 2]) + "\n")
    for ii in range(len(triangles)):
        fout.write("3 " + str(triangles[ii, 0]) + " " + str(triangles[ii, 1]) + " " + str(triangles[ii, 2]) + "\n")
    fout.close()


def write_ply_polygon(name, vertices, polygons):
    fout = open(name, 'w')
    fout.write("ply\n")
    fout.write("format ascii 1.0\n")
    fout.write("element vertex " + str(len(vertices)) + "\n")
    fout.write("property float x\n")
    fout.write("property float y\n")
    fout.write("property float z\n")
    fout.write("element face " + str(len(polygons)) + "\n")
    fout.write("property list uchar int vertex_index\n")
    fout.write("end_header\n")
    for ii in range(len(vertices)):
        fout.write(str(vertices[ii][0]) + " " + str(vertices[ii][1]) + " " + str(vertices[ii][2]) + "\n")
    for ii in range(len(polygons)):
        fout.write(str(len(polygons[ii])))
        for jj in range(len(polygons[ii])):
            fout.write(" " + str(polygons[ii][jj]))
        fout.write("\n")
    fout.close()


def read_ply_polygon(name):
    fin = open(name, 'r')
    content = fin.readlines()
    vertices_len = int(content[2][15:-1])
    polygons_len = int(content[6][13:-1])
    vertices = content[9:9 + vertices_len]
    polygons = content[9 + vertices_len: 9 + vertices_len + polygons_len]
    fin.close()
    temp_file = "current.txt"
    fout = open(temp_file, 'w')
    fout.writelines(vertices)
    fout.flush()
    vertices = np.loadtxt(temp_file)
    fout = open(temp_file, 'w')
    fout.writelines(polygons)
    fout.flush()
    polygons = np.loadtxt(temp_file).astype(np.int)
    fout.close()
    return vertices, polygons[:, 1:4]


def write_obj_triangle(name, vertices, triangles):
    fout = open(name, 'w')
    for ii in range(len(vertices)):
        fout.write("v " + str(vertices[ii, 0]) + " " + str(vertices[ii, 1]) + " " + str(vertices[ii, 2]) + "\n")
    for ii in range(len(triangles)):
        fout.write(
            "f " + str(triangles[ii, 0] + 1) + " " + str(triangles[ii, 1] + 1) + " " + str(triangles[ii, 2] + 1) + "\n")
    fout.close()


def write_obj_polygon(name, vertices, polygons):
    fout = open(name, 'w')
    for ii in range(len(vertices)):
        fout.write("v " + str(vertices[ii][0]) + " " + str(vertices[ii][1]) + " " + str(vertices[ii][2]) + "\n")
    for ii in range(len(polygons)):
        fout.write("f")
        for jj in range(len(polygons[ii])):
            fout.write(" " + str(polygons[ii][jj] + 1))
        fout.write("\n")
    fout.close()


# designed to take 64^3 voxels!
def sample_points_polygon_vox64(vertices, polygons, voxel_model_64, num_of_points):
    # convert polygons to triangles
    triangles = []
    for ii in range(len(polygons)):
        for jj in range(len(polygons[ii]) - 2):
            triangles.append([polygons[ii][0], polygons[ii][jj + 1], polygons[ii][jj + 2]])
    triangles = np.array(triangles, np.int32)
    vertices = np.array(vertices, np.float32)

    small_step = 1.0 / 64
    epsilon = 1e-6
    triangle_area_list = np.zeros([len(triangles)], np.float32)
    triangle_normal_list = np.zeros([len(triangles), 3], np.float32)
    for i in range(len(triangles)):
        # area = |u x v|/2 = |u||v|sin(uv)/2
        a, b, c = vertices[triangles[i, 1]] - vertices[triangles[i, 0]]
        x, y, z = vertices[triangles[i, 2]] - vertices[triangles[i, 0]]
        ti = b * z - c * y
        tj = c * x - a * z
        tk = a * y - b * x
        area2 = math.sqrt(ti * ti + tj * tj + tk * tk)
        if area2 < epsilon:
            triangle_area_list[i] = 0
            triangle_normal_list[i, 0] = 0
            triangle_normal_list[i, 1] = 0
            triangle_normal_list[i, 2] = 0
        else:
            triangle_area_list[i] = area2
            triangle_normal_list[i, 0] = ti / area2
            triangle_normal_list[i, 1] = tj / area2
            triangle_normal_list[i, 2] = tk / area2

    triangle_area_sum = np.sum(triangle_area_list)
    sample_prob_list = (num_of_points / triangle_area_sum) * triangle_area_list

    triangle_index_list = np.arange(len(triangles))

    point_normal_list = np.zeros([num_of_points, 6], np.float32)
    count = 0
    watchdog = 0

    while (count < num_of_points):
        np.random.shuffle(triangle_index_list)
        watchdog += 1
        if watchdog > 100:
            print("infinite loop here!")
            return point_normal_list
        for i in range(len(triangle_index_list)):
            if count >= num_of_points: break
            dxb = triangle_index_list[i]
            prob = sample_prob_list[dxb]
            prob_i = int(prob)
            prob_f = prob - prob_i
            if np.random.random() < prob_f:
                prob_i += 1
            normal_direction = triangle_normal_list[dxb]
            u = vertices[triangles[dxb, 1]] - vertices[triangles[dxb, 0]]
            v = vertices[triangles[dxb, 2]] - vertices[triangles[dxb, 0]]
            base = vertices[triangles[dxb, 0]]
            for j in range(prob_i):
                # sample a point here:
                u_x = np.random.random()
                v_y = np.random.random()
                if u_x + v_y >= 1:
                    u_x = 1 - u_x
                    v_y = 1 - v_y
                ppp = u * u_x + v * v_y + base

                # verify normal
                pppn1 = (ppp + normal_direction * small_step + 0.5) * 64
                px1 = int(pppn1[0])
                py1 = int(pppn1[1])
                pz1 = int(pppn1[2])

                ppx = int((ppp[0] + 0.5) * 64)
                ppy = int((ppp[1] + 0.5) * 64)
                ppz = int((ppp[2] + 0.5) * 64)

                if ppx < 0 or ppx >= 64 or ppy < 0 or ppy >= 64 or ppz < 0 or ppz >= 64:
                    continue
                if voxel_model_64[
                    ppx, ppy, ppz] > 1e-3 or px1 < 0 or px1 >= 64 or py1 < 0 or py1 >= 64 or pz1 < 0 or pz1 >= 64 or \
                        voxel_model_64[px1, py1, pz1] > 1e-3:
                    # valid
                    point_normal_list[count, :3] = ppp
                    point_normal_list[count, 3:] = normal_direction
                    count += 1
                    if count >= num_of_points: break

    return point_normal_list


def sample_points_polygon(vertices, polygons, num_of_points):
    # convert polygons to triangles
    triangles = []
    for ii in range(len(polygons)):
        for jj in range(len(polygons[ii]) - 2):
            triangles.append([polygons[ii][0], polygons[ii][jj + 1], polygons[ii][jj + 2]])
    triangles = np.array(triangles, np.int32)
    vertices = np.array(vertices, np.float32)

    small_step = 1.0 / 64
    epsilon = 1e-6
    triangle_area_list = np.zeros([len(triangles)], np.float32)
    triangle_normal_list = np.zeros([len(triangles), 3], np.float32)
    for i in range(len(triangles)):
        # area = |u x v|/2 = |u||v|sin(uv)/2
        a, b, c = vertices[triangles[i, 1]] - vertices[triangles[i, 0]]
        x, y, z = vertices[triangles[i, 2]] - vertices[triangles[i, 0]]
        ti = b * z - c * y
        tj = c * x - a * z
        tk = a * y - b * x
        area2 = math.sqrt(ti * ti + tj * tj + tk * tk)
        if area2 < epsilon:
            triangle_area_list[i] = 0
            triangle_normal_list[i, 0] = 0
            triangle_normal_list[i, 1] = 0
            triangle_normal_list[i, 2] = 0
        else:
            triangle_area_list[i] = area2
            triangle_normal_list[i, 0] = ti / area2
            triangle_normal_list[i, 1] = tj / area2
            triangle_normal_list[i, 2] = tk / area2

    triangle_area_sum = np.sum(triangle_area_list)
    sample_prob_list = (num_of_points / triangle_area_sum) * triangle_area_list

    triangle_index_list = np.arange(len(triangles))

    point_normal_list = np.zeros([num_of_points, 6], np.float32)
    count = 0
    watchdog = 0

    while (count < num_of_points):
        np.random.shuffle(triangle_index_list)
        watchdog += 1
        if watchdog > 100:
            print("infinite loop here!")
            return point_normal_list
        for i in range(len(triangle_index_list)):
            if count >= num_of_points: break
            dxb = triangle_index_list[i]
            prob = sample_prob_list[dxb]
            prob_i = int(prob)
            prob_f = prob - prob_i
            if np.random.random() < prob_f:
                prob_i += 1
            normal_direction = triangle_normal_list[dxb]
            u = vertices[triangles[dxb, 1]] - vertices[triangles[dxb, 0]]
            v = vertices[triangles[dxb, 2]] - vertices[triangles[dxb, 0]]
            base = vertices[triangles[dxb, 0]]
            for j in range(prob_i):
                # sample a point here:
                u_x = np.random.random()
                v_y = np.random.random()
                if u_x + v_y >= 1:
                    u_x = 1 - u_x
                    v_y = 1 - v_y
                point_normal_list[count, :3] = u * u_x + v * v_y + base
                point_normal_list[count, 3:] = normal_direction
                count += 1
                if count >= num_of_points: break

    return point_normal_list


def sample_points(vertices, triangles, num_of_points):
    epsilon = 1e-6
    triangle_area_list = np.zeros([len(triangles)], np.float32)
    triangle_normal_list = np.zeros([len(triangles), 3], np.float32)
    for i in range(len(triangles)):
        # area = |u x v|/2 = |u||v|sin(uv)/2
        a, b, c = vertices[triangles[i, 1]] - vertices[triangles[i, 0]]
        x, y, z = vertices[triangles[i, 2]] - vertices[triangles[i, 0]]
        ti = b * z - c * y
        tj = c * x - a * z
        tk = a * y - b * x
        area2 = math.sqrt(ti * ti + tj * tj + tk * tk)
        if area2 < epsilon:
            triangle_area_list[i] = 0
            triangle_normal_list[i, 0] = 0
            triangle_normal_list[i, 1] = 0
            triangle_normal_list[i, 2] = 0
        else:
            triangle_area_list[i] = area2
            triangle_normal_list[i, 0] = ti / area2
            triangle_normal_list[i, 1] = tj / area2
            triangle_normal_list[i, 2] = tk / area2

    triangle_area_sum = np.sum(triangle_area_list)
    sample_prob_list = (num_of_points / triangle_area_sum) * triangle_area_list

    triangle_index_list = np.arange(len(triangles))

    point_normal_list = np.zeros([num_of_points, 6], np.float32)
    count = 0
    watchdog = 0

    while (count < num_of_points):
        np.random.shuffle(triangle_index_list)
        watchdog += 1
        if watchdog > 100:
            print("infinite loop here!")
            exit(0)
        for i in range(len(triangle_index_list)):
            if count >= num_of_points: break
            dxb = triangle_index_list[i]
            prob = sample_prob_list[dxb]
            prob_i = int(prob)
            prob_f = prob - prob_i
            if np.random.random() < prob_f:
                prob_i += 1
            normal_direction = triangle_normal_list[dxb]
            u = vertices[triangles[dxb, 1]] - vertices[triangles[dxb, 0]]
            v = vertices[triangles[dxb, 2]] - vertices[triangles[dxb, 0]]
            base = vertices[triangles[dxb, 0]]
            for j in range(prob_i):
                # sample a point here:
                u_x = np.random.random()
                v_y = np.random.random()
                if u_x + v_y >= 1:
                    u_x = 1 - u_x
                    v_y = 1 - v_y
                point_normal_list[count, :3] = (u * u_x + v * v_y + base)
                point_normal_list[count, 3:] = normal_direction
                count += 1
                if count >= num_of_points: break

    return point_normal_list



def load_near_data(obj, cls):
    # pre_path = os.path.join(dataset_path, txt_list[idx])
    # train_points = np.load(os.path.join(pre_path, "points_train.npz"))
    # clean_pc = train_points["point_cloud"]
    obj = "04530566/194ea4b2297cc2ce90c91984b829ab51"
    # sdf = np.load("data/npy/"+ "/" + obj+"_gt.off.npy")
    # sdf = sdf[0]
    # sdf = np.clip(sdf, a_max=0.03, a_min=-0.03)
    mesh = trimesh.load("data/gt/"+obj+"_gt.off")
    points, face_idx = trimesh.sample.sample_surface_even(mesh, 3500)
    N = points.shape[0]
    if N < 3000:
        choice = np.random.choice(N, 3000 - N)
        points = np.concatenate([points, points[choice, :]], 0)
        face_idx = np.concatenate([face_idx, face_idx[choice]], 0)
    else:
        points = points[0:3000]
        face_idx = face_idx[0:3000]
    # normals = mesh.face_normals[face_idx]
    # np.savetxt("pc.txt", points, delimiter=";")

    clean_point, _ = trimesh.sample.sample_surface(mesh, 200000)
    # pc_near = clean_point + (np.random.random(clean_point.shape)-0.5)*0.01

    # pc_sdf, shift = cal_udf(mesh, pc_near)

    # ppp = np.where(np.abs(sdf)<0.01)
    # ppp = np.stack(ppp).transpose(1,0)
    # from udf_extractor import sdf_extractor
    # m2 = sdf_extractor(sdf)
    # np.savetxt("pc.txt", ppp/128.0-1.0, delimiter=";")
    # mesh.export("gt.off")
    # m2.export("rec.off")
    # exit(0)

    points = torch.from_numpy(points.astype(np.float32))
    # pc_sdf = torch.from_numpy(pc_sdf.astype(np.float32))
    clean_point = torch.from_numpy(clean_point.astype(np.float32))

    b_scale = points.max(0)[0] - points.min(0)[0]
    scale = ((b_scale[0] * b_scale[1] + b_scale[0] * b_scale[2] + b_scale[2]
              * b_scale[1]) / 2 * 4).sqrt()
    scale = torch.clamp(scale, min=1.0)

    loc = 0
    points = (points - loc) / scale
    clean_point = (clean_point - loc) / scale

    return points, clean_point, scale

def load_data(train_file=""):
    h5 = h5py.File(train_file)
    point_cloud = h5["point_cloud"][:].astype(np.float32)
    point_cloud2 = h5["point_cloud2"][:].astype(np.float32)
    sign = h5["sign"][:].astype(np.float32)
    return point_cloud, point_cloud2, sign

def density(pc):
    """
    :param pc:  BNC
    :return: B
    """
    knn_index, dis = knn_point(36, pc, pc, True)
    # mean_dis = dis.mean()
    return dis

def load_data_g(txt_list, dxb, dataset_path=""):
    point_cloud = []
    points_chamfer = []
    clean_point = []

    for idx in dxb:
        pre_path = os.path.join(dataset_path, txt_list[idx])
        pc = np.load(os.path.join(pre_path, "pointcloud.npz"), allow_pickle=False)

        code, uuid = txt_list[idx].split("/")
        mesh_path = "/disk2/occupancy_networks-master/data/ABC.build/001/4_watertight_scaled/" + \
                    uuid + ".off"
        mesh = trimesh.load(mesh_path, process=False)
        # npy_path = "data/npy/" +code +"/"+ uuid + ".npy"
        # mesh = trimesh.load("data/gt/"+code+"/" + uuid + "_gt.off", process=False)
        points= pc["points"]
        chamfer_choice = torch.randperm(points.shape[0])[0:100000].numpy()
        recon_choice = torch.randperm(points.shape[0])[0:100000].numpy()
        points_large = points[chamfer_choice]
        points = points[recon_choice]

        noise = 0.000 * np.random.randn(*points.shape)
        noise = noise.astype(np.float32)
        points_noise = points + noise

        point_cloud.append(points_noise)
        points_chamfer.append(points_large)
        train_points = np.load(os.path.join(pre_path, "points_train.npz"))
        clean_pc = train_points["point_cloud"]
        # clean_pc -= 0.5
        clean_point.append(clean_pc)
    point_cloud = np.stack(point_cloud).astype(np.float32)
    clean_point = np.stack(clean_point).astype(np.float32)
    points_chamfer = np.stack(points_chamfer).astype(np.float32)

    point_cloud = torch.from_numpy(point_cloud)[:, :, 0:3]
    clean_point = torch.from_numpy(clean_point)[:, :, 0:3]
    points_chamfer = torch.from_numpy(points_chamfer)[:, :, 0:3]
    # l = chamfer_loss(point_cloud, points_chamfer)
    # if os.path.exists(npy_path):
    #     npy = np.load(npy_path)
    # else:
    #     npy = None
    # vertices, triangles = mcubes.marching_cubes(-npy, 0.0)
    # vertices /= np.array([256, 256, 256])
    # vertices -= 0.5
    # mesh = trimesh.Trimesh(vertices, triangles, process=False)
    # mesh.export("sss.off")
    # np.savetxt("sss.txt", clean_point[0], delimiter=";")
    # exit(0)
    return point_cloud, clean_point, points_chamfer, mesh # torch.from_numpy(normal.astype(np.float32)).unsqueeze(0)



def load_data_test_iou(txt_list, dxb, dataset_path=""):
    point_cloud = []
    query_p = []
    query_l = []
    clean_point = []
    chosen_num = 10000
    for idx in dxb:
        pre_path = os.path.join(dataset_path, txt_list[idx])
        train_points = np.load(os.path.join(pre_path, "points_train.npz"))
        clean_pc, points_surface, points_uniform, occupancies_surface, occupancies_uniform = \
            train_points["point_cloud"], train_points["points_surface"], train_points["points_uniform"], train_points[
                "occupancies_surface"], \
            train_points["occupancies_uniform"]
        clean_pc -= 0.5

        pc = np.load(os.path.join(pre_path, "pointcloud.npz"))
        points = pc["points"]
        chamfer_choice = torch.randperm(points.shape[0])[0:3000].numpy()
        recon_choice = torch.randperm(points.shape[0])[0:3000].numpy()
        points_large = points[chamfer_choice]
        points = points[recon_choice]
        # indices = np.random.randint(points.shape[0], size=3000)
        # points = points[indices, :]
        noise = 0.005 * np.random.randn(*points.shape)
        noise = noise.astype(np.float32)
        points_noise = points + noise
        clean_point.append(points)
        point_cloud.append(points_noise)
        train_points = np.load(os.path.join(pre_path, "points.npz"))
        points, occupancies = train_points["points"], np.unpackbits(train_points["occupancies"])
        query_p.append(points)
        query_l.append(occupancies)
    point_cloud = np.stack(point_cloud).astype(np.float32)
    clean_point = np.stack(clean_point).astype(np.float32)
    query_p = np.stack(query_p).astype(np.float32)
    query_l = np.stack(query_l).astype(np.float32)
    # np.savetxt("gt.txt", query_p[0][np.where(query_l[0]>0.5)], delimiter=";")
    # np.savetxt("pc.txt", point_cloud[0], delimiter=";")
    # print(txt_list[dxb[0]])
    # exit(0)
    query_p = torch.from_numpy(query_p)
    query_l = torch.from_numpy(query_l)
    point_cloud = torch.from_numpy(point_cloud)[:, :, 0:3]
    clean_point = torch.from_numpy(clean_point)[:, :, 0:3]
    return point_cloud, query_p, query_l, clean_point


def load_data_test(txt_list, dxb, dataset_path=""):
    point_cloud = []
    loc_l = []
    scale_l = []
    clean_point = []
    chosen_num = 10000
    for idx in dxb:
        pre_path = os.path.join(dataset_path, txt_list[idx])
        pc = np.load(os.path.join(pre_path, "pointcloud.npz"))
        train_points = np.load(os.path.join(pre_path, "points_train.npz"))

        clean_pc, points_surface, points_uniform, occupancies_surface, occupancies_uniform = \
            train_points["point_cloud"], train_points["points_surface"], train_points["points_uniform"], train_points[
                "occupancies_surface"], \
            train_points["occupancies_uniform"]
        clean_pc -= 0.5

        loc_l.append(pc["loc"])
        points = pc["points"]
        point_choice = torch.randperm(points.shape[0])[0:3000].numpy()
        points = points[point_choice, :]
        noise = 0.000 * np.random.randn(*points.shape)
        noise = noise.astype(np.float32)
        points_noise = points + noise
        clean_point.append(points.astype(np.float32))
        point_cloud.append(points_noise)
    point_cloud = np.stack(point_cloud)
    clean_point = np.stack(clean_point)
    loc_l = np.stack(loc_l)
    point_cloud = torch.from_numpy(point_cloud)
    clean_point = torch.from_numpy(clean_point)
    loc_l = torch.from_numpy(loc_l)
    return point_cloud, clean_point, loc_l

def chamfer_loss(points_src, points_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    dist_matrix = ((points_src.unsqueeze(2) - points_tgt.unsqueeze(1))**2).sum(-1) # B M T
    dist_complete = (dist_matrix.min(-1)[0]).mean(-1)
    dist_acc = (dist_matrix.min(-2)[0]).mean(-1)
    dist = ((dist_acc*0 + dist_complete*2)/2).mean()*1e4
    return dist

def chamfer_loss_chunk_no_grad(points_src, points_tgt, trainig=True):
    ''' Computes minimal distances of each point in points_src to points_tgt.
    tgt chunk
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    with torch.no_grad():
        B, N, _ = points_src.shape
        chunk_size = 200
        G = points_tgt.shape[1]
        chunk_num = math.ceil(G/chunk_size)
        point_tgt_list = torch.chunk(points_tgt, chunk_num, 1)
        dist_complete_list = []
        dist_complete_index_list = []
        dist_acc_list = []
        for p_tgt in point_tgt_list:
            dist_matrix = ((points_src.unsqueeze(2) - p_tgt.unsqueeze(1)) ** 2).sum(-1)  # B N T
            rest = dist_matrix.min(-1)
            dist_complete_list.append(rest[0])  # B N 1
            dist_complete_index_list.append(rest[1])
            dist_acc_list.append((dist_matrix.min(-2)[1]))  # B 1 C
        target_closest_index = torch.stack(dist_acc_list).permute(1, 0, 2).reshape(B, -1)  # B T

        # R B N
        row_index = torch.stack(dist_complete_list).min(0)[1]
        BNR = torch.stack(dist_complete_index_list).permute(1,2,0).reshape(B*N, -1)  # B N R

        BN = BNR[torch.arange(B*N), row_index.reshape(-1)].reshape(B, N)

        src_cloest_index = row_index*chunk_size + BN
        src_cloest_point = index_points(points_tgt, src_cloest_index)
        target_closest_point = index_points(points_src, target_closest_index)
    if trainig:
        dist_complete = ((points_src - src_cloest_point)**2).sum(-1).mean()
        dist_acc = ((points_tgt - target_closest_point) ** 2).sum(-1).mean()
        dist = ((dist_acc*0 + dist_complete*2)/2)*1e4
    else:
        dist_complete = ((points_src - src_cloest_point).abs()).sum(-1).mean()
        dist_acc = ((points_tgt - target_closest_point).abs()).sum(-1).mean()
        dist = ((dist_acc * 1 + dist_complete * 1) / 2)

    return dist, points_src - src_cloest_point



def chamfer_loss_chunk_local(points_src, points_tgt, trainig=True):
    ''' Computes minimal distances of each point in points_src to points_tgt.
    tgt chunk
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    with torch.no_grad():
        B, N, _ = points_src.shape
        chunk_size = 500
        G = points_tgt.shape[1]
        chunk_num = math.ceil(G/chunk_size)
        point_tgt_list = torch.chunk(points_tgt, chunk_num, 1)
        dist_complete_list = []
        dist_complete_index_list = []
        dist_acc_list = []
        for p_tgt in point_tgt_list:
            dist_matrix = ((points_src.unsqueeze(2) - p_tgt.unsqueeze(1)) ** 2).sum(-1)  # B N T
            rest = dist_matrix.min(-1)
            dist_complete_list.append(rest[0])  # B N 1
            dist_complete_index_list.append(rest[1])
            dist_acc_list.append((dist_matrix.min(-2)[1]))  # B 1 C
        target_closest_index = torch.stack(dist_acc_list).permute(1, 0, 2).reshape(B, -1)  # B T

        # R B N
        row_index = torch.stack(dist_complete_list).min(0)[1]
        BNR = torch.stack(dist_complete_index_list).permute(1,2,0).reshape(B*N, -1)  # B N R

        BN = BNR[torch.arange(B*N), row_index.reshape(-1)].reshape(B, N)

        src_cloest_index = row_index*chunk_size + BN
        src_cloest_point = index_points(points_tgt, src_cloest_index)
        target_closest_point = index_points(points_src, target_closest_index)

        dist_complete = ((points_src - src_cloest_point)**2).sum(-1).mean(-1)
        dist_acc = ((points_tgt - target_closest_point)**2).sum(-1).mean(-1)
        dist = ((dist_acc * 1 + dist_complete * 1) / 2)*1e4

    return dist, points_src - src_cloest_point


def chamfer_loss_chunk_no_grad_closest(origin_point, points_src, points_tgt, scale=1.0):
    ''' Computes minimal distances of each point in points_src to points_tgt.
    tgt chunk
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    with torch.no_grad():
        B, N, _ = points_src.shape
        chunk_size = 1000
        G = points_tgt.shape[1]
        chunk_num = math.ceil(G/chunk_size)
        point_tgt_list = torch.chunk(points_tgt, chunk_num, 1)
        dist_complete_list = []
        dist_complete_index_list = []
        dist_acc_list = []
        for p_tgt in point_tgt_list:
            dist_matrix = ((points_src.unsqueeze(2) - p_tgt.unsqueeze(1)) ** 2).sum(-1)  # B N T
            rest = dist_matrix.min(-1)
            dist_complete_list.append(rest[0])  # B N 1
            dist_complete_index_list.append(rest[1])
            dist_acc_list.append((dist_matrix.min(-2)[1]))  # B 1 C
        target_closest_index = torch.stack(dist_acc_list).permute(1, 0, 2).reshape(B, -1)  # B T

        # R B N
        row_index = torch.stack(dist_complete_list).min(0)[1]
        BNR = torch.stack(dist_complete_index_list).permute(1,2,0).reshape(B*N, -1)  # B N R

        BN = BNR[torch.arange(B*N), row_index.reshape(-1)].reshape(B, N)

        src_cloest_index = row_index*chunk_size + BN
        src_cloest_point = index_points(points_tgt, src_cloest_index)
        target_closest_point = index_points(points_src, target_closest_index)
    dist_complete = ((points_src - src_cloest_point)**2).sum(-1)+1e-7
    dist_complete = (dist_complete*(scale.unsqueeze(1).unsqueeze(1)**2).to(dist_complete.device)).mean()
    # dist_acc = ((points_tgt - target_closest_point).abs()).sum(-1)+1e-7
    # dist_acc = (dist_acc*(scale.unsqueeze(1).unsqueeze(1)).to(dist_acc.device)).mean()
    dist = ((0 + dist_complete)*2/2)*1e4
    return dist, src_cloest_point - origin_point



def chamfer_loss_chunk_no_grad_weighted(origin_point, points_src, points_tgt, scale=1.0):
    ''' Computes minimal distances of each point in points_src to points_tgt.
    tgt chunk
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): gt points
        normals_tgt (numpy array): target normals
    '''
    # ll = chamfer_loss(points_src, points_tgt)

    B, N, _ = points_src.shape
    knn_num = 8
    chamfer_l = 0
    for iii, p_tgt in enumerate(points_tgt):
        tree = KDTree(p_tgt.detach().cpu().numpy())
        _, ng_idx = tree.query(points_src[iii].detach().cpu().numpy(), k=knn_num)
        dist, _ = tree.query(origin_point[iii].detach().cpu().numpy(), k=knn_num)
        ng_idx = ng_idx.astype(np.int64)
        weight = F.softmax(torch.from_numpy(-dist*400).unsqueeze(1), dim=-1).to(points_src.device)
        neighbor_point = index_points(p_tgt.unsqueeze(0), torch.from_numpy(ng_idx).unsqueeze(0))[0]
        # neighbor_point = p_tgt[ng_idx]
        chamfer = (((neighbor_point - points_src[iii].unsqueeze(1))**2).sum(-1)*weight).sum(-1).mean()
        # ((p_tgt[idx] - points_src[iii])** 2).sum(-1)
        chamfer_l += chamfer*scale.to(points_src.device)[iii]**2
    chamfer_l = chamfer_l*10000/B
    return chamfer_l, chamfer_l


def chamfer_loss_chunk_no_grad_withnrom(origin_point, points_src, points_tgt, orientation):
    ''' Computes minimal distances of each point in points_src to points_tgt.
    tgt chunk
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    with torch.no_grad():
        B, N, _ = points_src.shape
        chunk_size = 500
        G = points_tgt.shape[1]
        chunk_num = math.ceil(G/chunk_size)
        point_tgt_list = torch.chunk(points_tgt, chunk_num, 1)
        dist_complete_list = []
        dist_complete_index_list = []
        # dist_acc_list = []
        for p_tgt in point_tgt_list:
            dist_matrix = ((origin_point.unsqueeze(2) - p_tgt.unsqueeze(1)) ** 2).sum(-1)  # B N T
            rest = dist_matrix.min(-1)
            dist_complete_list.append(rest[0])  # B N 1
            dist_complete_index_list.append(rest[1])
            # dist_acc_list.append((dist_matrix.min(-2)[1]))  # B 1 C
        # target_closest_index = torch.stack(dist_acc_list).permute(1, 0, 2).reshape(B, -1)  # B T

        # R B N
        row_index = torch.stack(dist_complete_list).min(0)[1]
        BNR = torch.stack(dist_complete_index_list).permute(1,2,0).reshape(B*N, -1)  # B N R

        BN = BNR[torch.arange(B*N), row_index.reshape(-1)].reshape(B, N)

        src_cloest_index = row_index*chunk_size + BN
        src_cloest_point = index_points(points_tgt, src_cloest_index)
        src_cloest_normal = index_points(orientation, src_cloest_index)
        dist_complete = ((src_cloest_normal*(points_src - src_cloest_point)).sum(-1)**2).mean()
        # target_closest_point = index_points(points_src, target_closest_index)
        dist_complete2 = ((points_src - src_cloest_point)**2).sum(-1).mean()

        dist = ((0 + dist_complete*2)/2)*1e4
    return dist, src_cloest_point - origin_point

def chamfer_loss_chunk(points_src, points_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.
    tgt chunk
    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    chunk_size = 1000
    G = points_tgt.shape[1]
    chunk_num = math.ceil(G/chunk_size)
    point_tgt_list = torch.chunk(points_tgt, chunk_num, 1)
    dist_complete_list = []
    dist_acc_list = []
    for p_tgt in point_tgt_list:
        dist_matrix = ((points_src.unsqueeze(2) - p_tgt.unsqueeze(1)) ** 2).sum(-1)
        dist_complete_list.append(dist_matrix.min(-1)[0])
        dist_acc_list.append((dist_matrix.min(-2)[0]).mean(-1))
    dist_acc_list = torch.stack(dist_acc_list).mean()
    dist_complete_list = torch.stack(dist_complete_list).min(0)[0].mean()
    dist = ((dist_acc_list*0 + dist_complete_list*1)/1)*1e4
    return dist

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            knn_num = 24
            dists, idx = dists[:, :, :knn_num], idx[:, :, :knn_num]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, knn_num, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points