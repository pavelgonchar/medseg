import caffe
from caffe import layers as L, params as P
# enum Engine {DEFAULT = 0;CAFFE = 1;CUDNN = 2;}
# enum PoolMethod { MAX = 0; AVE = 1; STOCHASTIC = 2;}
# enum Phase {TRAIN = 0;TEST = 1;}
# weight_filler=dict(type='xavier') or = dict(type='gaussian', std=0.001)
# bias_filler=dict(type='constant', value=0)
'''Example:
# Data:
L.Input(input_param=dict(shape=dict(dim=[2, 1, 321, 321])))

# Convolution for equal kernel_size:
L.Convolution(bottom,
	num_output=64, pad=1, kernel_size=3, stride=1,
	weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
	param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
	engine=1)
# Convolution for unequal kernel_size:
L.Convolution(bottom,
	param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
	num_output=64, pad=[1,3,3], kernel_size=[3,7,7], stride=2,
	weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
	engine=1)

# Deconvolution:
L.Deconvolution(bottom,
	convolution_param=dict(num_output=21, kernel_size=64, stride=32, bias_term=False),
	param=[dict(lr_mult=0)])
L.Deconvolution(bottom,
	param=[dict(lr_mult=1, decay_mult=1)],
	convolution_param=dict(num_output=128,  pad=0, kernel_size=[2,2,1], stride=[2,2,1],
		weight_filler=dict(type='gaussian', std=0.001), bias_term=0, engine=1))

# BN:
if phase == "TRAIN":
	L.BatchNorm(bottom, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
else:
	L.BatchNorm(bottom, use_global_stats=1)

# Scale:
L.Scale(bottom, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))

# ReLU:
L.ReLU(bottom, in_place=True, engine=1)

# Pool:
Max:
L.Pooling(bottom, pool=0, kernel_size=3, stride=2, pad=1, engine=1)
AVE:
L.Pooling(bottom, pool=1, kernel_size=3, stride=2, pad=1, engine=1)

# NDPool:
Max:
L.PoolingND(bottom, pool=0, kernel_size=2, stride=2, engine=1)
AVE:
L.PoolingND(bottom, pool=1, kernel_size=[3,5,5], stride=1, engine=1)
# Dropout:
L.Dropout(bottom, dropout_ratio=0.5, in_place=True)
# Crop:
L.Crop(bottom1, bottom2, axis=2, offset=0)
# Concat:
L.Concat(bottom1, bottom2, axis=1)

# SoftmaxWithLoss:
L.SoftmaxWithLoss(bottom, label, phase=0, loss_weight=1, loss_param=dict(ignore_label=ignore_label))
L.SoftmaxWithLoss(n.score, n.label, loss_param=dict(normalize=False, ignore_label=255))
# Softmax:
L.Softmax(bottom, phase=1)
'''

############ ############
def conv_relu_all(bottom,
	num_output,
	bias_term=True,
	pad=0, kernel_size=3, stride=1,
	dilation=1,
	weight_filler=dict(type='xavier'),
	bias_filler=dict(type='constant', value=0),
	param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
	engine=0
	):
	conv = L.Convolution(bottom,
		num_output=num_output,
		bias_term=bias_term,
		pad=pad, kernel_size=kernel_size, stride=stride,
		dilation=dilation,
		weight_filler=weight_filler,
		bias_filler=bias_filler,
		engine=engine,
		param=param
		)
	relu = L.ReLU(conv,
		engine=engine,
		in_place=True)
	return conv, relu
############ ############
def conv_bn_scale_relu_all(bottom,
	num_output,
	bias_term=True,
	pad=0, kernel_size=3, stride=1,
	dilation=1,
	weight_filler=dict(type='xavier'),
	bias_filler=dict(type='constant', value=0),
	param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
	engine=0,
	phase='train'
	):
	conv = L.Convolution(bottom,
		num_output=num_output,
		bias_term=bias_term,
		pad=pad, kernel_size=kernel_size, stride=stride,
		dilation=dilation,
		weight_filler=weight_filler,
		bias_filler=bias_filler,
		engine=engine,
		param=param
		)
	if phase == 'train':
		bn = L.BatchNorm(conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		bn = L.BatchNorm(conv, use_global_stats=1)
	scale = L.Scale(bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	relu = L.ReLU(scale,
		engine=engine,
		in_place=True)
	return conv, bn, scale, relu
############ ############
def conv_relu(bottom, num_output, pad=0, kernel_size=3, stride=1):
    conv = L.Convolution(bottom,
    	num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
    	# weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0),
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    relu = L.ReLU(conv, in_place=True)
    return conv, relu
############ ############
def conv_bn_scale_relu(bottom, num_output, pad=0, kernel_size=3, stride=1, phase='train'):
	conv = L.Convolution(bottom,
		num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
		# weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0)
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
	if phase == 'train':
		bn = L.BatchNorm(conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		bn = L.BatchNorm(conv, use_global_stats=1)
	scale = L.Scale(bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	relu = L.ReLU(scale, in_place=True)
	return conv, bn, scale, relu
############ ############
def max_pool(bottom, pad=0, kernel_size=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, pad=pad, kernel_size=kernel_size, stride=stride)
############ ############
def ave_pool(bottom, pad=0, kernel_size=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.AVE, pad=pad, kernel_size=kernel_size, stride=stride)
############ ############
def max_pool_nd(bottom, pad=[0,0,0], kernel_size=[2,2,2], stride=[2,2,2]):
	return L.PoolingND(bottom, pool=0, pad=pad, kernel_size=kernel_size, stride=stride, engine=1)
############ ############
def ave_pool_nd(bottom, pad=[0,0,0], kernel_size=[2,2,2], stride=[2,2,2]):
	return L.PoolingND(bottom, pool=1, pad=pad, kernel_size=kernel_size, stride=stride, engine=1)
