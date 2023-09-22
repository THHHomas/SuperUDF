import time

import torch
from scipy.optimize import leastsq
import mcubes
import trimesh
import numpy as np
import torch.nn.functional as F



def sdf_extractor(data):
    # data = F.pad(torch.from_numpy(data), (1, 1, 1, 1, 1, 1), 'constant', 1e6).numpy()
    vertices, triangles = mcubes.marching_cubes(-data, 0.0)
    # vertices += 0.5
    vertices /= data.shape
    vertices -= 0.5
    mesh = trimesh.Trimesh(vertices, triangles, process=False)
    return mesh


def generate_grid():
    grid = []
    for x in [0, 1, 2]:
        for y in [0, 1, 2]:
            for z in [0, 1, 2]:
                grid.append(np.array([x**2,y**2,z**2,x*y,y*z,x*z,x, y, z, 1]))
    grid = np.stack(grid)
    return grid


def conflict_detect(sign):
    """
    :param sign:
    :return: True means conflict
    """
    vertex_sgin = np.array([0]*8)
    # init vertex sign
    if sign[0]:
        vertex_sgin[0] = 1
        vertex_sgin[1] = -1
    else:
        vertex_sgin[0] = 1
        vertex_sgin[1] = 1

    if sign[4]:
        vertex_sgin[4] = 1
        vertex_sgin[5] = -1
    else:
        vertex_sgin[4] = 1
        vertex_sgin[5] = 1

    if sign[2]:
        vertex_sgin[2] = 1
        vertex_sgin[3] = -1
    else:
        vertex_sgin[2] = 1
        vertex_sgin[3] = 1

    if sign[6]:
        vertex_sgin[6] = 1
        vertex_sgin[7] = -1
    else:
        vertex_sgin[6] = 1
        vertex_sgin[7] = 1
    # return False, vertex_sgin
    # detect along z axis
    if sign[3]:
        if vertex_sgin[0]*vertex_sgin[3] > 0:
            return True, None
    else:
        if vertex_sgin[0]*vertex_sgin[3] < 0:
            return True, None

    if sign[7]:
        if vertex_sgin[4] * vertex_sgin[7] > 0:
            return True, None
    else:
        if vertex_sgin[4] * vertex_sgin[7] < 0:
            return True, None

    if sign[1]:
        if vertex_sgin[1] * vertex_sgin[2] > 0:
            return True, None
    else:
        if vertex_sgin[1] * vertex_sgin[2] < 0:
            return True, None

    if sign[5]:
        if vertex_sgin[5] * vertex_sgin[6] > 0:
            return True, None
    else:
        if vertex_sgin[5] * vertex_sgin[6] < 0:
            return True, None

    # detect along y axis
    if sign[8]:
        if vertex_sgin[0] * vertex_sgin[4] > 0:
            return True, None
    else:
        if vertex_sgin[0] * vertex_sgin[4] < 0:
            return True, None

    if sign[9]:
        if vertex_sgin[1] * vertex_sgin[5] > 0:
            return True, None
    else:
        if vertex_sgin[5] * vertex_sgin[1] < 0:
            return True, None

    if sign[10]:
        if vertex_sgin[2] * vertex_sgin[6] > 0:
            return True, None
    else:
        if vertex_sgin[2] * vertex_sgin[6] < 0:
            return True, None

    if sign[11]:
        if vertex_sgin[3] * vertex_sgin[7] > 0:
            return True, None
    else:
        if vertex_sgin[3] * vertex_sgin[7] < 0:
            return True, None

    return False, vertex_sgin


def VertexInterp(isolevel,p1,p2,valp1,valp2):
    p = np.zeros(3)
    if (abs(isolevel-valp1) < 1e-5):
        return p1
    if (abs(isolevel-valp2) < 1e-5):
        return p2
    if (abs(valp1-valp2) < 1e-5):
        return p1
    mu = (isolevel - valp1) / (valp2 - valp1)
    p[0] = p1[0] + mu * (p2[0] - p1[0])
    p[1] = p1[1] + mu * (p2[1] - p1[1])
    p[2] = p1[2] + mu * (p2[2] - p1[2])
    return p


def get_vertex(cubeindex, grid_pos, grid_val, isolevel=0):
    edgeTable = [
        0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
        0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
        0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
        0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
        0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
        0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
        0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
        0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
        0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
        0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
        0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
        0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
        0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
        0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
        0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
        0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
        0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
        0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
        0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
        0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
        0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
        0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
        0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
        0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
        0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
        0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
        0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
        0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
        0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
        0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
        0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
        0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0]
    vertlist = [0]*12
    if (edgeTable[cubeindex] // 1)%2:
        vertlist[0] = VertexInterp(isolevel, grid_pos[0], grid_pos[1], grid_val[0], grid_val[1])
    if (edgeTable[cubeindex] // 2)%2:
        vertlist[1] = VertexInterp(isolevel, grid_pos[1], grid_pos[2], grid_val[1], grid_val[2])
    if (edgeTable[cubeindex] // 4)%2:
        vertlist[2] =VertexInterp(isolevel, grid_pos[2], grid_pos[3], grid_val[2], grid_val[3])
    if (edgeTable[cubeindex] // 8)%2:
        vertlist[3] =VertexInterp(isolevel, grid_pos[3], grid_pos[0], grid_val[3], grid_val[0])
    if (edgeTable[cubeindex] // 16)%2:
        vertlist[4] =VertexInterp(isolevel, grid_pos[4], grid_pos[5], grid_val[4], grid_val[5])
    if (edgeTable[cubeindex] // 32)%2:
        vertlist[5] =VertexInterp(isolevel, grid_pos[5], grid_pos[6], grid_val[5], grid_val[6])
    if (edgeTable[cubeindex] // 64)%2:
        vertlist[6] =VertexInterp(isolevel, grid_pos[6], grid_pos[7], grid_val[6], grid_val[7])
    if (edgeTable[cubeindex] // 128)%2:
        vertlist[7] =VertexInterp(isolevel, grid_pos[7], grid_pos[4], grid_val[7], grid_val[4])
    if (edgeTable[cubeindex] // 256)%2:
        vertlist[8] =VertexInterp(isolevel, grid_pos[0], grid_pos[4], grid_val[0], grid_val[4])
    if (edgeTable[cubeindex] // 512)%2:
        vertlist[9] =VertexInterp(isolevel, grid_pos[1], grid_pos[5], grid_val[1], grid_val[5])
    if (edgeTable[cubeindex] // 1024)%2:
        vertlist[10] =VertexInterp(isolevel, grid_pos[2], grid_pos[6], grid_val[2], grid_val[6])
    if (edgeTable[cubeindex] // 2048)%2:
        vertlist[11] =VertexInterp(isolevel, grid_pos[3], grid_pos[7], grid_val[3], grid_val[7])
    return vertlist


def generate_ver_tri(ver_value, left_up_index, past_grid, vertice_num, isolevel=0):
    """
    :param ver_sign:  1, -1
    :return:
    """
    ver_value = ver_value.reshape(-1)
    ver_value[2], ver_value[3] = ver_value[3], ver_value[2]
    ver_value[6], ver_value[7] = ver_value[7], ver_value[6]

    ver_sign = (ver_value<isolevel).astype(np.int)
    cube_index = 0
    for idx, s in enumerate(ver_sign):
        cube_index += s*(2**idx)
    grid_pos = [[0,0,0], [0,0,1], [0,1,1], [0,1,0], [1,0,0], [1,0,1], [1,1,1], [1,1,0]]

    # grid_pos = np.array(grid_pos)
    triTable = [
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1],
        [3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1],
        [3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
        [3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1],
        [9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1],
        [9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
        [2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1],
        [8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1],
        [9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
        [4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1],
        [3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1],
        [1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1],
        [4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1],
        [4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1],
        [9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
        [5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1],
        [2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1],
        [9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1],
        [0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1],
        [2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1],
        [10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1],
        [4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1],
        [5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1],
        [5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1],
        [9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1],
        [0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1],
        [1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1],
        [10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1],
        [8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1],
        [2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1],
        [7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1],
        [9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1],
        [2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1],
        [11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1],
        [9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1],
        [5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1],
        [11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1],
        [11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
        [1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1],
        [9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1],
        [5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1],
        [2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1],
        [5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1],
        [6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1],
        [3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1],
        [6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1],
        [5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1],
        [1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1],
        [10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1],
        [6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1],
        [8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1],
        [7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1],
        [3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1],
        [5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1],
        [0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1],
        [9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1],
        [8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1],
        [5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1],
        [0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1],
        [6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1],
        [10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1],
        [10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1],
        [8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1],
        [1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1],
        [3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1],
        [0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1],
        [10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1],
        [3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1],
        [6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1],
        [9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1],
        [8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1],
        [3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1],
        [6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1],
        [0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1],
        [10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1],
        [10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1],
        [2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1],
        [7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1],
        [7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1],
        [2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1],
        [1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1],
        [11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1],
        [8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1],
        [0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1],
        [7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
        [10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
        [2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1],
        [6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1],
        [7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1],
        [2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1],
        [1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1],
        [10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1],
        [10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1],
        [0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1],
        [7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1],
        [6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1],
        [8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1],
        [9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1],
        [6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1],
        [4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1],
        [10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1],
        [8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1],
        [0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1],
        [1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1],
        [8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1],
        [10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1],
        [4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1],
        [10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1],
        [5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
        [11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1],
        [9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1],
        [6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1],
        [7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1],
        [3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1],
        [7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1],
        [9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1],
        [3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1],
        [6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1],
        [9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1],
        [1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1],
        [4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1],
        [7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1],
        [6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1],
        [3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1],
        [0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1],
        [6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1],
        [0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1],
        [11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1],
        [6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1],
        [5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1],
        [9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1],
        [1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1],
        [1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1],
        [10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1],
        [0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1],
        [5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1],
        [10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1],
        [11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1],
        [9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1],
        [7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1],
        [2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1],
        [8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1],
        [9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1],
        [9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1],
        [1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1],
        [9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1],
        [9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1],
        [5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1],
        [0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1],
        [10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1],
        [2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1],
        [0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1],
        [0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1],
        [9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1],
        [5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1],
        [3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1],
        [5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1],
        [8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1],
        [0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1],
        [9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1],
        [0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1],
        [1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1],
        [3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1],
        [4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1],
        [9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1],
        [11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1],
        [11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1],
        [2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1],
        [9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1],
        [3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1],
        [1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1],
        [4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1],
        [4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1],
        [3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1],
        [3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1],
        [0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1],
        [9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1],
        [1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    ]
    vertex = get_vertex(cube_index, grid_pos, ver_value)

    edge = triTable[cube_index]
    triangle = []
    triangles = []
    vertices = []
    pair = {0:[0,0],3:[0,0],8:[0,0],
            2:[0,1],9:[0,1],1:[0,1],
            6:[1,1],10:[1,1],5:[1,1],
            4:[1,0],11:[1,0],7:[1,0]}
    past_index = {0:0,2:0,4:0,6:0,
                  8:1,9:1,10:1,11:1,
                  1:2,3:2,5:2,7:2}
    for idx, local_edge_index in enumerate(edge):
        if local_edge_index == -1:
            break
        current_x, current_y, current_z = left_up_index.astype(np.int64)
        edge_index = pair[local_edge_index]
        if past_index[local_edge_index] == 2:
            index = past_grid[0, current_x+edge_index[0], current_y, current_z+edge_index[1]]
        if past_index[local_edge_index] == 1:
            index = past_grid[1, current_x, current_y+edge_index[0], current_z+edge_index[1]]
        if past_index[local_edge_index] == 0:
            index = past_grid[2, current_x+edge_index[0], current_y+edge_index[1], current_z]

        if index == -1:
            ######################  add new vertex
            vertices.append(vertex[local_edge_index]+left_up_index)
            triangle.append(vertice_num)
            if past_index[local_edge_index] == 2:
                past_grid[0, current_x + edge_index[0], current_y, current_z + edge_index[1]] = vertice_num
            elif past_index[local_edge_index] == 1:
                past_grid[1, current_x, current_y + edge_index[0], current_z + edge_index[1]] = vertice_num
            elif past_index[local_edge_index] == 0:
                past_grid[2, current_x + edge_index[0], current_y + edge_index[1], current_z] = vertice_num
            vertice_num += 1
        else:
            ################# add old vertex to triangles
            triangle.append(index)
        if idx % 3 == 2:
            triangles.append(np.array(triangle))
            triangle = []
    return np.array(triangles).astype(np.int64), np.array(vertices), past_grid, vertice_num


def udf_extractor(data, sign_data):
    new_data = np.zeros_like(data)
    eps = 1.5e-2
    dim = data.shape
    A = generate_grid()
    vertices_list = []
    triangles_list = []
    vertice_num = 0
    for i_idx in range(dim[0]//2):
        for j_idx in range(dim[1]//2):
            for k_idx in range(dim[2]//2):
                current_grid = data[i_idx*2:i_idx*2 + 3, j_idx*2:j_idx*2 + 3, k_idx*2:k_idx*2 + 3]
                min_udf = current_grid.min()
                if min_udf > 0.03:
                    continue
                confidence = np.zeros(12, dtype=np.float)

                confidence[0] = min((current_grid[0, 0, 0] - current_grid[1, 0, 0]),
                    (current_grid[2, 0, 0] - current_grid[1, 0, 0]))
                confidence[1] = min((current_grid[2, 0, 0] - current_grid[2, 0, 1]), (
                        current_grid[2, 0, 2] - current_grid[2, 0, 1]))
                confidence[2] = min((current_grid[0, 0, 2] - current_grid[1, 0, 2]) , (
                        current_grid[2, 0, 2] - current_grid[1, 0, 2]))
                confidence[3] = min((current_grid[0, 0, 0] - current_grid[0, 0, 1]) , (
                        current_grid[0, 0, 2] - current_grid[0, 0, 1]))

                confidence[4] = min((current_grid[0, 2, 0] - current_grid[1, 2, 0]), (
                        current_grid[2, 2, 0] - current_grid[1, 2, 0]))
                confidence[5] = min((current_grid[2, 2, 0] - current_grid[2, 2, 1]), (
                        current_grid[2, 2, 2] - current_grid[2, 2, 1]))
                confidence[6] = min((current_grid[0, 2, 2] - current_grid[1, 2, 2]), (
                        current_grid[2, 2, 2] - current_grid[1, 2, 2]))
                confidence[7] = min((current_grid[0, 2, 0] - current_grid[0, 2, 1]), (
                        current_grid[0, 2, 2] - current_grid[0, 2, 1]))

                confidence[8] = min((current_grid[0, 0, 0] - current_grid[0, 1, 0]), (
                        current_grid[0, 2, 0] - current_grid[0, 1, 0]))
                confidence[9] = min((current_grid[2, 0, 0] - current_grid[2, 1, 0]), (
                        current_grid[2, 2, 0] - current_grid[2, 1, 0]))
                confidence[10] = min((current_grid[2, 0, 2] - current_grid[2, 1, 2]), (
                        current_grid[2, 2, 2] - current_grid[2, 1, 2]))
                confidence[11] = min((current_grid[0, 0, 2] - current_grid[0, 1, 2]), (
                        current_grid[0, 2, 2] - current_grid[0, 1, 2]))
                sign = confidence > 0
                confidence = np.abs(confidence)
                # B = data[i_idx:i_idx+3, j_idx:j_idx+3, k_idx:k_idx+3].reshape(-1)
                # ATA = A.T.dot(A)
                # ATb = A.T.dot(B)
                # inv_ATA = np.linalg.inv(ATA)
                # solution = inv_ATA.dot(ATb)
                # predict = A.dot(solution)
                # a, b, c, d, e, f, g, h, i, j = solution
                # x_axis = []
                # y_axis = []
                # z_axis = []
                # for x1 in [0, 2]:
                #     for x2 in [0, 2]:
                #         if a<1e-5:
                #             x_axis.append(np.array([1, x1, x2]))
                #         else:
                #             x_axis.append(np.array([-(d*x1+f*x2+g)/(2*a), x1, x2]))
                #
                #         if b<1e-5:
                #             y_axis.append(np.array([x1, 1, x2]))
                #         else:
                #             y_axis.append(np.array([x1, -(d*x1+e*x2+h)/(2*b), x2]))
                #
                #         if c<1e-5:
                #             z_axis.append(np.array([x1, x2, 1]))
                #         else:
                #             z_axis.append(np.array([x1, x2, -(f*x1+e*x2+i)/(2*c)]))
                # xyz = np.concatenate([x_axis, y_axis, z_axis], 0).T
                # grid = np.stack([xyz[0] ** 2, xyz[1] ** 2, xyz[2] ** 2, xyz[0] * xyz[1], xyz[1] * xyz[2], xyz[0] * xyz[2], xyz[0], xyz[1], xyz[2], np.ones_like(xyz[2])]).T
                # alpha = grid.dot(solution)
                # sign_index = [0,2,4,6,8,11,9,10,3,7,1,5]
                # new_alpha = np.zeros(12)
                # new_alpha[sign_index] = alpha
                # sign = (alpha < eps)

                # confidence = np.abs(alpha - eps)
                index = np.argsort(confidence)
                chosen_idx = 0

                while True:
                    conflict, vertex_sgin = conflict_detect(sign)
                    # break
                    if not conflict or chosen_idx == 11:
                        break
                    flip_index = index[chosen_idx]
                    sign[flip_index] = not sign[flip_index]
                    chosen_idx += 1
                vertex_sgin = np.sign(sign_data[i_idx*2:i_idx*2 + 3:2, j_idx*2:j_idx*2 + 3:2, k_idx*2:k_idx*2 + 3:2])\
                    .reshape(-1).astype(np.int)
                if np.random.random() > 0.5:
                    vertex_sgin = vertex_sgin*-1

                vertex_value = vertex_sgin.reshape(2,2,2)*data[i_idx*2:i_idx*2 + 3:2, j_idx*2:j_idx*2 + 3:2, k_idx*2:k_idx*2 + 3:2]

                new_data[i_idx:i_idx + 2, j_idx:j_idx + 2, k_idx:k_idx + 2] = vertex_sgin.reshape(2,2,2)\
                                                                              *data[i_idx*2:i_idx*2 + 3:2, j_idx*2:j_idx*2 + 3:2, k_idx*2:k_idx*2 + 3:2]

                if True:
                    triangles, vertices = generate_ver_tri(vertex_value)
                    if len(vertices)>0:
                        triangles += vertice_num
                        vertice_num += len(vertices)
                        vertices_list.append(vertices)
                        vertices += np.expand_dims(np.array([i_idx*2, j_idx*2, k_idx*2]), axis=0)
                        # generate_ver_tri(vertex_sgin)

                        triangles_list.append(triangles)
    triangles_list = np.concatenate(triangles_list, 0)
    vertices_list = np.concatenate(vertices_list, 0)
    vertices_list -= 1
    vertices_list = vertices_list.astype(np.float32)
    vertices_list /= data.shape
    vertices_list -= 0.5
    mesh = trimesh.Trimesh(vertices_list, triangles_list, process=False)
    return mesh, new_data

if __name__ == '__main__':

    data = np.load("../samples/bsp_ae_out/udf_data/test/0_udf.npy")
    # data=data[0]
    # data = data[::4, ::4, ::4]
    # data = data[126:135, 106:115, 126:135]
    # data = np.sign(data)
    s = time.time()
    # mesh, new_data = udf_extractor(np.abs(data), data)
    mesh_sdf = sdf_extractor(-data[0])
    print(time.time()-s)
    mesh_sdf.export("0.off")
    # mesh_sdf.export("data/1b00f29471a41f59e92b1dc10fc46551.off")
    ss = 0

