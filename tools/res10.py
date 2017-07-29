import _init_paths
import os
import os.path as osp
from lits.config import cfg
import caffe
from caffe import layers as L, params as P, to_proto

def UNet3DBN(input_dims, class_nums, phase="TRAIN"):
	net = caffe.NetSpec()

	############ d0 ############
	### a ###
	net.data = L.Input(input_param=dict(shape=dict(dim=input_dims)))
	if phase == "TRAIN":
		net.label = L.Input(input_param=dict(shape=dict(dim=input_dims)))
		net.label_weight = L.Input(input_param=dict(shape=dict(dim=input_dims)))
	### b ###
	net.d0b_conv = L.Convolution(net.data,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=32, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.d0b_bn = L.BatchNorm(net.d0b_conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d0b_bn = L.BatchNorm(net.d0b_conv, use_global_stats=1)
	net.d0b_scale = L.Scale(net.d0b_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d0b_relu = L.ReLU(net.d0b_scale, in_place=True, engine=1)
	### c ###
	net.d0c_conv = L.Convolution(net.d0b_scale,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=32, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.d0c_bn = L.BatchNorm(net.d0c_conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d0c_bn = L.BatchNorm(net.d0c_conv, use_global_stats=1)
	net.d0c_scale = L.Scale(net.d0c_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d0c_relu = L.ReLU(net.d0c_scale, in_place=True, engine=1)

	
	############ d1 ############
	### a ### First pooling
	net.d1a_pool = L.PoolingND(net.d0c_scale,
		pool=0,
		kernel_size=[2,2,1],
		stride=[2,2,1],
		engine=1)
	### b ###
	net.d1b_conv = L.Convolution(net.d1a_pool,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=64, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN": 
		net.d1b_bn = L.BatchNorm(net.d1b_conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d1b_bn = L.BatchNorm(net.d1b_conv, use_global_stats=1)
	net.d1b_scale = L.Scale(net.d1b_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d1b_relu = L.ReLU(net.d1b_scale, in_place=True, engine=1)
	### c ###
	net.d1c_conv = L.Convolution(net.d1b_scale,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=64, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.d1c_bn = L.BatchNorm(net.d1c_conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d1c_bn = L.BatchNorm(net.d1c_conv, use_global_stats=1)
	net.d1c_scale = L.Scale(net.d1c_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d1c_relu = L.ReLU(net.d1c_scale, in_place=True, engine=1)


	############ d2 ############
	### a ###
	net.d2a_pool = L.PoolingND(net.d1c_scale,
		pool=0,
		kernel_size=2,
		stride=2,
		engine=1)
	### b ###
	net.d2b_conv = L.Convolution(net.d2a_pool,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=128, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.d2b_bn = L.BatchNorm(net.d2b_conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d2b_bn = L.BatchNorm(net.d2b_conv, use_global_stats=1)
	net.d2b_scale = L.Scale(net.d2b_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d2b_relu = L.ReLU(net.d2b_scale, in_place=True, engine=1)
	### c ###
	net.d2c_conv = L.Convolution(net.d2b_scale,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=128, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.d2c_bn = L.BatchNorm(net.d2c_conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d2c_bn = L.BatchNorm(net.d2c_conv, use_global_stats=1)
	net.d2c_scale = L.Scale(net.d2c_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d2c_relu = L.ReLU(net.d2c_scale, in_place=True, engine=1)


	############ d3 ############
	### a ### Third Pooling
	net.d3a_pool = L.PoolingND(net.d2c_scale,
		pool=0,
		kernel_size=2,
		stride=2,
		engine=1)
	### b ###
	net.d3b_conv = L.Convolution(net.d3a_pool,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=256, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.d3b_bn = L.BatchNorm(net.d3b_conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d3b_bn = L.BatchNorm(net.d3b_conv, use_global_stats=1)
	net.d3b_scale = L.Scale(net.d3b_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d3b_relu = L.ReLU(net.d3b_scale, in_place=True, engine=1)
	### c ###
	net.d3c_conv = L.Convolution(net.d3b_scale,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=256, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.d3c_bn = L.BatchNorm(net.d3c_conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.d3c_bn = L.BatchNorm(net.d3c_conv, use_global_stats=1)
	net.d3c_scale = L.Scale(net.d3c_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.d3c_relu = L.ReLU(net.d3c_scale, in_place=True, engine=1)


	############ u2 ############
	### a ### First Deconvolution
	net.u2a_dconv = L.Deconvolution(net.d3c_scale,
		param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=128,  pad=0, kernel_size=2, stride=2,
			weight_filler=dict(type='gaussian', std=0.001), bias_term=0,
			engine=1))
	if phase == "TRAIN":
		net.u2a_bn = L.BatchNorm(net.u2a_dconv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.u2a_bn = L.BatchNorm(net.u2a_dconv, use_global_stats=1)
	net.u2a_scale = L.Scale(net.u2a_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.u2a_relu = L.ReLU(net.u2a_scale, in_place=True, engine=1)
	### b ### Crop and Concat
	net.u2b_crop = L.Crop(net.u2a_scale, net.d2c_scale, axis=2, offset=0)
	net.u2b_concat = L.Concat(net.u2b_crop, net.d2c_scale, axis=1)
	### c ###
	net.u2c_conv = L.Convolution(net.u2b_concat,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=128, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.u2c_bn = L.BatchNorm(net.u2c_conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.u2c_bn = L.BatchNorm(net.u2c_conv, use_global_stats=1)
	net.u2c_scale = L.Scale(net.u2c_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.u2c_relu = L.ReLU(net.u2c_scale, in_place=True, engine=1)
	### d ###
	net.u2d_conv = L.Convolution(net.u2c_scale,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=128, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.u2d_bn = L.BatchNorm(net.u2d_conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.u2d_bn = L.BatchNorm(net.u2d_conv, use_global_stats=1)
	net.u2d_scale = L.Scale(net.u2d_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.u2d_relu = L.ReLU(net.u2d_scale, in_place=True, engine=1)


	############ u1 ############
	### a ### Second Deconvolution
	net.u1a_dconv = L.Deconvolution(net.u2d_scale,
		param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=64,  pad=0, kernel_size=2, stride=2,
			weight_filler=dict(type='gaussian', std=0.001), bias_term=0,
			engine=1))
	if phase == "TRAIN":
		net.u1a_bn = L.BatchNorm(net.u1a_dconv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.u1a_bn = L.BatchNorm(net.u1a_dconv, use_global_stats=1)
	net.u1a_scale = L.Scale(net.u1a_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.u1a_relu = L.ReLU(net.u1a_scale, in_place=True, engine=1)
	### b ### Crop and Concat
	net.u1b_crop = L.Crop(net.u1a_scale, net.d1c_scale, axis=2, offset=0)
	net.u1b_concat = L.Concat(net.u1b_crop, net.d1c_scale, axis=1)
	### c ###
	net.u1c_conv = L.Convolution(net.u1b_concat,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=64, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.u1c_bn = L.BatchNorm(net.u1c_conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.u1c_bn = L.BatchNorm(net.u1c_conv, use_global_stats=1)
	net.u1c_scale = L.Scale(net.u1c_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.u1c_relu = L.ReLU(net.u1c_scale, in_place=True, engine=1)
	### d ###
	net.u1d_conv = L.Convolution(net.u1c_scale,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=64, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.u1d_bn = L.BatchNorm(net.u1d_conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.u1d_bn = L.BatchNorm(net.u1d_conv, use_global_stats=1)
	net.u1d_scale = L.Scale(net.u1d_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.u1d_relu = L.ReLU(net.u1d_scale, in_place=True, engine=1)


	############ u0 ############
	### a ### Third Deconvolution
	net.u0a_dconv = L.Deconvolution(net.u1d_scale,
		param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=32,  pad=0, kernel_size=[2,2,1], stride=[2,2,1],
			weight_filler=dict(type='gaussian', std=0.001), bias_term=0,
			engine=1))
	if phase == "TRAIN": 
		net.u0a_bn = L.BatchNorm(net.u0a_dconv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.u0a_bn = L.BatchNorm(net.u0a_dconv, use_global_stats=1)
	net.u0a_scale = L.Scale(net.u0a_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.u0a_relu = L.ReLU(net.u0a_scale, in_place=True, engine=1)

	### b ### Crop and Concat
	net.u0b_crop = L.Crop(net.u0a_scale, net.d0c_scale, axis=2, offset=0)
	net.u0b_concat = L.Concat(net.u0b_crop, net.d0c_scale, axis=1)

	### c ###
	net.u0c_conv = L.Convolution(net.u0b_concat,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=32, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN": 
		net.u0c_bn = L.BatchNorm(net.u0c_conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.u0c_bn = L.BatchNorm(net.u0c_conv, use_global_stats=1)
	net.u0c_scale = L.Scale(net.u0c_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.u0c_relu = L.ReLU(net.u0c_scale, in_place=True, engine=1)
	### d ###
	net.u0d_conv = L.Convolution(net.u0c_scale,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=32, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)
	if phase == "TRAIN":
		net.u0d_bn = L.BatchNorm(net.u0d_conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		net.u0d_bn = L.BatchNorm(net.u0d_conv, use_global_stats=1)
	net.u0d_scale = L.Scale(net.u0d_bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	net.u0d_relu = L.ReLU(net.u0d_scale, in_place=True, engine=1)


	############ Score ############
	net.u0d_score = L.Convolution(net.u0d_scale,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=class_nums, pad=1, kernel_size=3, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)

	############ Loss ############
	if phase == "TRAIN":
		net.loss = L.WeightedSoftmaxWithLoss(net.u0d_score, net.label, net.label_weight,
			phase=0,
			loss_weight=1,
			loss_param=dict(ignore_label=255))

	return net.to_proto()