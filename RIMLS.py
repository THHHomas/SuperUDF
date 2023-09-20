import numpy as np
import open3d as o3d
import pcl
# 读取点云数据
points= np.loadtxt("data/output/ling_64/0_inputpc.txt", delimiter=";")

pcl_cloud = pcl.PointCloud()
pcl_cloud.from_array(points.astype(np.float32))

# 创建RIMLS对象
rimls = pcl.RobustImplicitMLS(pcl_cloud)
# 设置参数
rimls.setUpsamplingMethod(pcl.RobustImplicitMLS.MovingLeastSquares)
rimls.setUpsamplingRadius(0.01)
rimls.setUpsamplingStepSize(0.005)
# 执行重建
reconstructed = rimls.reconstruct()


voxel_filter = reconstructed.make_voxel_grid_filter()
voxel_filter.set_leaf_size(0.01, 0.01, 0.01)
downsampled = voxel_filter.filter()

# 可视化结果
viewer = pcl.visualization.CloudViewing()
viewer.ShowMonochromeCloud(downsampled)
while not viewer.WasStopped():
    pass