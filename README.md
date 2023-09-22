# BSP-NET-pytorch
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
  We provide the processed dataset of ScanNet in [data.zip](https://pan.baidu.com/s/1CXPiXAbaW4gavEMjDs4RTg) (pwd: 1234). 
After download the dataset, you should put it in `./data`.


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

of course, you can use the pretrained weight in `data/checkpoint256.pth`.




