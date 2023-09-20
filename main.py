import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np

from modelAE import BSP_AE

import argparse
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("--phase", action="store", dest="phase", default=1, type=int, help="phase 0 = continuous, phase 1 = hard discrete, phase 2 = hard discrete with L_overlap, phase 3 = soft discrete, phase 4 = soft discrete with L_overlap [1]")
#phase 0 continuous for better convergence
#phase 1 hard discrete for bsp
#phase 2 hard discrete for bsp with L_overlap
#phase 3 soft discrete for bsp
#phase 4 soft discrete for bsp with L_overlap
#use [phase 0 -> phase 1] or [phase 0 -> phase 2] or [phase 0 -> phase 3] or [phase 0 -> phase 4]
parser.add_argument("--epoch", action="store", dest="epoch", default=0, type=int, help="Epoch to train [0]")
parser.add_argument("--iteration", action="store", dest="iteration", default=0, type=int, help="Iteration to train. Either epoch or iteration need to be zero [0]")
parser.add_argument("--learning_rate", action="store", dest="learning_rate", default=0.001, type=float, help="Learning rate for adam [0.0001]")
parser.add_argument("--beta1", action="store", dest="beta1", default=0.5, type=float, help="Momentum term of adam [0.5]")
parser.add_argument("--dataset", action="store", dest="dataset", default="all_vox256_img", help="The name of dataset")
parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="checkpoint", help="Directory name to save the checkpoints [checkpoint]")
parser.add_argument("--data_dir", action="store", dest="data_dir", default=".", help="Root directory of dataset [data]")
parser.add_argument("--sample_dir", action="store", dest="sample_dir", default="./samples/", help="Directory name to save the image samples [samples]")
parser.add_argument("--sample_vox_size", action="store", dest="sample_vox_size", default=64, type=int, help="Voxel resolution for coarse-to-fine training [64]")
parser.add_argument("--train", action="store_true", dest="train", default=False, help="True for training, False for testing [False]")
parser.add_argument("--start", action="store", dest="start", default=0, type=int, help="In testing, output shapes [start:end]")
parser.add_argument("--end", action="store", dest="end", default=-1, type=int, help="In testing, output shapes [start:end]")
parser.add_argument("--ae", action="store_true", dest="ae", default=False, help="True for ae [False]")
parser.add_argument("--denoise", action="store_true", dest="denoise", default=False, help="True for denoise [False]")
parser.add_argument("--gpu", action="store", dest="gpu", default="1", help="Which GPU to use [0]")
FLAGS = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu


if not os.path.exists(FLAGS.sample_dir):
	os.makedirs(FLAGS.sample_dir) 

bsp_ae = BSP_AE(FLAGS)
if FLAGS.train:
	bsp_ae.train(FLAGS)
else:
	bsp_ae.test_dae3(FLAGS)


