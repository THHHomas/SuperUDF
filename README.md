# SuperUDF
PyTorch 1.4.0 implementation of [SuperUDF: Self-supervised UDF Estimation for Surface Reconstruction](https://arxiv.org/abs/2308.14371).


## Citation
If you find our work useful in your research, please consider citing:

	@article{superudf,
	  title={SuperUDF: Self-supervised UDF Estimation for Surface Reconstruction},
	  author={Hui Tian, Chenyang Zhu, Yifei Shi and Kai Xu.},
	  journal={IEEE Transactions on Visualization and Computer Graphics},
	  year={2023}
	}


## Dependencies
Requirements:
- Python 3.6 with numpy, h5py and Cython
- PyTorch 1.4
- [PyMCubes](https://github.com/pmneila/PyMCubes) (for marching cubes)
- TriMesh 3.12.9

## Datasets
  We provide the processed dataset of ScanNet in [data.zip](https://pan.baidu.com/s/1tBNYh9NQSnil7QigrNPqaQ) (pwd: qa8p). 
After download the dataset, you should unzip it and put it in `./data/test.h5`.


## UDF Prediction

we can train the UDF prediction network using the following commad,
```
python main.py --ae --train --phase 1 --iteration 300000 --dataset data/data_per_category/data_per_category/04530566_vessel/04530566_vox256_img --sample_dir data/output/vessel_64 --sample_vox_size 64
```
Next, we can generate upsampled point cloud and UDF data,
```
python main.py --ae --phase 1 --sample_dir samples/bsp_ae_out --dataset data/data_per_category/data_per_category/04530566_vessel/04530566_vox256_img --start 0 --end 100
```
The generated data is in `samples/bsp_ae_out/udf_data`.

## Mesh Extraction
After UDF prediction, we should first train the mesh extraction to generate mesh,
```cd mesh_extract```
```python tiny_udf_net.py --train```
Next, we should generate mesh with the pretrained network, 

```python tiny_udf_net.py```

the generated mesh is in `samples/bsp_ae_out/udf_data/test/*.off`. When you visualize the mesh in Meshlab, you should choose Back-Face module as Double and Shading module as Face, because the generated mesh normal is not consistent as most UDF-based method.

of course, you can use the pretrained weight in `data/checkpoint256.pth`.

 How to generate the `data/test.h5` is still a problem. In short, we randomly sample 10000 points on the mesh and obtain the sign of SDF according to the normal orientation. Later, we will add the code.
