import open3d as o3d
import torch
import os
import numpy as np

def recon(mesh_path, method):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    bbox = mesh.get_axis_aligned_bounding_box()
    scale = (bbox.max_bound - bbox.min_bound).max()
    mesh.translate(-mesh.get_center())
    mesh.scale(1 / scale * 0.9, mesh.get_center())


    #

    if method=='psr':
        print('run Poisson surface reconstruction')
        mesh.compute_triangle_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=30000, use_triangle_normal=True)
        # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=10)
    elif method =='bpa':
        print('run BPA')
        pcd = mesh.sample_points_uniformly(number_of_points=30000)
        points = np.asarray(pcd.points)
        np.savetxt("5_inputpc.txt", points,  delimiter=";")
        exit(0)
        new_points = np.random.rand(*points.shape)*0.003 + points
        pcd.points = o3d.utility.Vector3dVector(new_points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(0.1, 128))
        radii = [0.005, 0.002, 0.001, 0.05, 0.01, 0.02, 0.0005, 0.1]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
    # mesh.compute_vertex_normals()
    # mesh.paint_uniform_color([229/255,231/255,1.0])
    # o3d.visualization.draw_geometries([mesh])
    return mesh


def poisson_test(path, source_data_path, original_meth_path, method='psr'):
    cls_list = [""]
    for cls in cls_list:
        result_path = path+cls
        if not os.path.exists(result_path):
            os.mkdir(result_path)

        idx_list = []
        for ff in os.listdir(source_data_path+cls):
            if not os.path.isdir(source_data_path+cls+"/"+ff):
                idx, suffix = ff.split("_")
                if suffix == "shiftpc.txt":
                    idx_list.append(eval(idx))
        idx_list = sorted(idx_list)[0:51]
        for i, idx in enumerate(idx_list):
            idx = 5
            mesh_path = original_meth_path + cls + str(idx) +"_gt.off"
            mesh = recon(mesh_path, method=method)
            o3d.io.write_triangle_mesh(result_path+"/"+str(idx)+"_"+method+".off", mesh)
            break


if __name__ == '__main__':
    # poisson_test("samples/bsp_ae_out/001/PSR")
    # bpa_test("samples/bsp_ae_out/001/BPA")
    # poisson_test("data/PSR/", source_data_path="/disk2/unsupvised scan/samples/bsp_ae_out/",
    #              original_meth_path="/disk2/unsupvised scan/data/gt/test/", method='psr')
    poisson_test("data/BPA/", source_data_path="/disk2/unsupvised scan/samples/bsp_ae_out/",
                 original_meth_path="/disk2/unsupvised scan/data/gt/test/", method='bpa')
    # poisson_test("data/BPA_1000/", source_data_path="/disk2/unsupvised scan/samples/bsp_ae_out/",
    #              original_meth_path="/disk2/unsupvised scan/data/gt/test/", method='bpa')
    # poisson_test("data/PSR_1000/", source_data_path="/disk2/unsupvised scan/samples/bsp_ae_out/",
    #              original_meth_path="/disk2/unsupvised scan/data/gt/test/", method='psr')

