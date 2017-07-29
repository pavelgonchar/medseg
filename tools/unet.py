import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop # crop(net.upscore, net.data) automatically calculate the axis and offset

def eltwize_relu(bottom1, bottom2):
	residual_eltwise = L.Eltwise(bottom1, bottom2, eltwise_param=dict(operation=1))
	residual_eltwise_relu = L.ReLU(residual_eltwise, in_place=True, engine=engine)
	
	return residual_eltwise, residual_eltwise_relu

###
def rcu(bottom, num_output, pad=0, kernel_size=3, stride=1):
	conv1 = L.Convolution(bottom,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
		num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
		#weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=engine)

	relu1 = L.ReLU(conv1, in_place=True, engine=engine)

	conv2 = L.Convolution(conv1,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
		num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
		#weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=engine)

	eltw, relu2 = eltwize_relu(bottom, conv2)

	return conv1, relu1, conv2, eltw, relu2
###
def rcu_bn(bottom, num_output, pad=0, kernel_size=3, stride=1,phase='train'):
	conv1 = L.Convolution(bottom,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
		num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
		#weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=engine)
	if phase == 'train':
		bn1 = L.BatchNorm(conv1, in_place=True, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		bn1 = L.BatchNorm(conv1, in_place=True, use_global_stats=1)
	scale1 = L.Scale(conv1, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	relu1 = L.ReLU(conv1, in_place=True, engine=engine)

	conv2 = L.Convolution(conv1,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
		num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
		#weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=engine)
	if phase == 'train':
		bn2 = L.BatchNorm(conv2, in_place=True, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		bn2 = L.BatchNorm(conv2, in_place=True, use_global_stats=1)
	scale2 = L.Scale(conv2, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))

	eltw, relu2 = eltwize_relu(bottom, conv2)

	return conv1, bn1, scale1, relu1, conv2, bn2, scale2, eltw, relu2

############ ############
def conv_relu(bottom, num_output, pad=0, kernel_size=3, stride=1):
	conv = L.Convolution(bottom,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
		num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
		#weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=engine)
	relu = L.ReLU(conv, in_place=True, engine=engine)
	return conv, relu
############ ############
def conv_bn_scale_relu(bottom, num_output, pad=0, kernel_size=3, stride=1, phase='train'):
	conv = L.Convolution(bottom,
		num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
		# weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], engine=engine)
	if phase == 'train':
		bn = L.BatchNorm(conv, in_place=True, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		bn = L.BatchNorm(conv, in_place=True, use_global_stats=1)
	scale = L.Scale(bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	relu = L.ReLU(scale, in_place=True, engine=engine)
	# relu = L.PReLU(scale, in_place=True)
	return conv, bn, scale, relu
############ ############
def dila_conv_bn_scale_relu(bottom, num_output, pad=0, kernel_size=3, stride=1, dilation=1, phase='train',engine=1):
	conv = L.Convolution(bottom,
		num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
		dilation=dilation,
		# weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
		engine=engine)
	if phase == 'train':
		bn = L.BatchNorm(conv, in_place=True, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		bn = L.BatchNorm(conv, in_place=True, use_global_stats=1)
	scale = L.Scale(conv, axis=1, in_place=True, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	relu = L.ReLU(conv, in_place=True,engine=engine)
	return conv, bn, scale, relu
############ ############
def deconv_relu(bottom, num_output, pad=0, kernel_size=3, stride=1):
	deconv = L.Deconvolution(bottom,
		param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=num_output,  pad=pad, kernel_size=kernel_size, stride=stride,
			#weight_filler=dict(type='gaussian', std=0.001), bias_term=0,
			weight_filler=dict(type='msra'), bias_term=0,
			engine=engine))
	relu = L.ReLU(deconv, in_place=True, engine=engine)
	return deconv, relu
############ ############
def deconv_bn_scale_relu(bottom, num_output, pad=0, kernel_size=3, stride=1, phase='train'):
	deconv = L.Deconvolution(bottom,
		param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=num_output,  pad=pad, kernel_size=kernel_size, stride=stride,
			# weight_filler=dict(type='gaussian', std=0.001), bias_term=0,
			weight_filler=dict(type='msra'), bias_term=0,
			engine=engine))
	if phase == 'train':
		bn = L.BatchNorm(deconv, in_place=True, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		bn = L.BatchNorm(deconv, in_place=True, use_global_stats=1)
	scale = L.Scale(bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	relu = L.ReLU(scale, in_place=True, engine=engine)
	# relu = L.PReLU(scale, in_place=True)
	return deconv, bn, scale, relu
############ ############
def max_pool(bottom, pad=0, kernel_size=2, stride=2):
	return L.Pooling(bottom, pool=P.Pooling.MAX, pad=pad, kernel_size=kernel_size, stride=stride, engine=engine)
############ ############
def ave_pool(bottom, pad=0, kernel_size=2, stride=2):
	return L.Pooling(bottom, pool=P.Pooling.AVE, pad=pad, kernel_size=kernel_size, stride=stride, engine=engine)
############ ############
def max_pool_nd(bottom, pad=[0,0,0], kernel_size=[2,2,2], stride=[2,2,2]):
	return L.PoolingND(bottom, pool=0, pad=pad, kernel_size=kernel_size, stride=stride, engine=1)
############ ############
def ave_pool_nd(bottom, pad=[0,0,0], kernel_size=[2,2,2], stride=[2,2,2]):
	return L.PoolingND(bottom, pool=1, pad=pad, kernel_size=kernel_size, stride=stride, engine=1)


def unet_2d_bn_mask(dim_data, dim_label, num_class, ignore_label=255, phase="train"):
	net = caffe.NetSpec()
	############ d0 ############
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if phase == "train":
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		# net.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))	
	net.d0b_conv, net.d0b_bn, net.d0b_scale, net.d0b_relu = conv_bn_scale_relu(net.data, 64, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d0c_conv, net.d0c_bn, net.d0c_scale, net.d0c_relu = conv_bn_scale_relu(net.d0b_conv, 64, pad=1, kernel_size=3, stride=1, phase=phase)
	############ d1 ############
	net.d1a_pool = max_pool(net.d0c_conv, pad=0, kernel_size=2, stride=2)
	net.d1b_conv, net.d1b_bn, net.d1b_scale, net.d1b_relu = conv_bn_scale_relu(net.d1a_pool, 128, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d1c_conv, net.d1c_bn, net.d1c_scale, net.d1c_relu = conv_bn_scale_relu(net.d1b_conv, 128, pad=1, kernel_size=3, stride=1, phase=phase)
	############ d2 ############
	net.d2a_pool = max_pool(net.d1c_conv, pad=0, kernel_size=2, stride=2)
	net.d2b_conv, net.d2b_bn, net.d2b_scale, net.d2b_relu = conv_bn_scale_relu(net.d2a_pool, 256, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d2c_conv, net.d2c_bn, net.d2c_scale, net.d2c_relu = conv_bn_scale_relu(net.d2b_conv, 256, pad=1, kernel_size=3, stride=1, phase=phase)
	############ d3 ############
	net.d3a_pool = max_pool(net.d2c_conv, pad=0, kernel_size=2, stride=2)
	net.d3b_conv, net.d3b_bn, net.d3b_scale, net.d3b_relu = conv_bn_scale_relu(net.d3a_pool, 512, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d3c_conv, net.d3c_bn, net.d3c_scale, net.d3c_relu = conv_bn_scale_relu(net.d3b_conv, 512, pad=1, kernel_size=3, stride=1, phase=phase)
	############ d4 ############
	net.d4a_pool = max_pool(net.d3c_conv, pad=0, kernel_size=2, stride=2)
	net.d4b_conv, net.d4b_bn, net.d4b_scale, net.d4b_relu = conv_bn_scale_relu(net.d4a_pool, 1024, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d4c_conv, net.d4c_bn, net.d4c_scale, net.d4c_relu = conv_bn_scale_relu(net.d4b_conv, 1024, pad=1, kernel_size=3, stride=1, phase=phase)

	############ u3 ############
	### a ### First Deconvolution
	net.u3a_dconv, net.u3a_bn, net.u3a_scale, net.u3a_relu = deconv_bn_scale_relu(net.d4c_conv, 512, pad=0, kernel_size=2, stride=2, phase=phase)
	### b ### Crop and Concat
	#net.u3b_crop = L.Crop(net.u3a_dconv, net.d3c_conv, axis=2, offset=0)
	#net.u3b_crop = crop(net.u3a_relu, net.d3c_relu)
	net.u3b_concat = L.Concat(net.u3a_dconv, net.d3c_conv, axis=1)
	### c ###
	net.u3c_conv, net.u3c_bn, net.u3c_scale, net.u3c_relu = conv_bn_scale_relu(net.u3b_concat, 512, pad=1, kernel_size=3, stride=1, phase=phase)
	### d ###
	net.u3d_conv, net.u3d_bn, net.u3d_scale, net.u3d_relu = conv_bn_scale_relu(net.u3c_conv, 512, pad=1, kernel_size=3, stride=1, phase=phase)

	############ u2 ############
	### a ### Second Deconvolution
	net.u2a_dconv, net.u2a_bn, net.u2a_scale, net.u2a_relu = deconv_bn_scale_relu(net.u3d_conv, 256, pad=0, kernel_size=2, stride=2, phase=phase)
	### b ### Crop and Concat
	#net.u2b_crop = L.Crop(net.u2a_dconv, net.d2c_conv, axis=2, offset=0)
	#net.u2b_crop = crop(net.u2a_relu, net.d2c_relu)
	net.u2b_concat = L.Concat(net.u2a_dconv, net.d2c_conv, axis=1)
	### c ###
	net.u2c_conv, net.u2c_bn, net.u2c_scale, net.u2c_relu = conv_bn_scale_relu(net.u2b_concat, 256, pad=1, kernel_size=3, stride=1, phase=phase)
	### d ###
	net.u2d_conv, net.u2d_bn, net.u2d_scale, net.u2d_relu = conv_bn_scale_relu(net.u2c_conv, 256, pad=1, kernel_size=3, stride=1, phase=phase)

	############ u1 ############
	### a ### Third Deconvolution
	net.u1a_dconv, net.u1a_bn, net.u1a_scale, net.u1a_relu = deconv_bn_scale_relu(net.u2d_relu, 128, pad=0, kernel_size=2, stride=2, phase=phase)
	### b ### Crop and Concat
	#net.u1b_crop = L.Crop(net.u1a_dconv, net.d1c_conv, axis=2, offset=0)
	#net.u1b_crop = crop(net.u1a_relu, net.d1c_relu)
	net.u1b_concat = L.Concat(net.u1a_dconv, net.d1c_conv, axis=1)
	### c ###
	net.u1c_conv, net.u1c_bn, net.u1c_scale, net.u1c_relu = conv_bn_scale_relu(net.u1b_concat, 128, pad=1, kernel_size=3, stride=1, phase=phase)
	### d ###
	net.u1d_conv, net.u1d_bn, net.u1d_scale, net.u1d_relu = conv_bn_scale_relu(net.u1c_conv, 128, pad=1, kernel_size=3, stride=1, phase=phase)

	

	# ############ Score ############
	# net.score = L.Convolution(net.u0d_conv,
	# 	param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
	# 	num_output=num_class, pad=0, kernel_size=1, stride=1, 
	# 	# weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
	# 	weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
	# 	engine=engine)

	# ############ Loss ############
	# if phase == "train":
	# 	net.loss = L.SoftmaxWithLoss(net.score, net.label,
	# 		phase=0,
	# 		loss_weight=1,
	# 		loss_param=dict(ignore_label=ignore_label))
	# 	# net.loss = L.WeightedSoftmaxWithLoss(net.score, net.label, net.label_weight,
	# 	# 	phase=0,
	# 	# 	loss_weight=1,
	# 	# 	loss_param=dict(ignore_label=ignore_label))

	# ### loss 4 
	# net.score4 = L.Convolution(net.d4c_conv,
	# 	param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
	# 	num_output=num_class, pad=0, kernel_size=1, stride=1, 
	# 	weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	# net.upscore4 = L.Deconvolution(net.score4,
	# 	param=[dict(lr_mult=1, decay_mult=1)],
	# 	convolution_param=dict(num_output=num_class,  pad=8, kernel_size=32, stride=16,
	# 		weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	# if phase == "train":
	# 	net.loss4 = L.SoftmaxWithLoss(net.upscore4, net.label,
	# 		phase=0,
	# 		loss_weight=0.0625,
	# 		loss_param=dict(ignore_label=ignore_label))

	### loss 3
	net.score3 = L.Convolution(net.u3d_conv,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore3 = L.Deconvolution(net.score3,
		param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=num_class,  pad=4, kernel_size=16, stride=8,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss3 = L.SoftmaxWithLoss(net.upscore3, net.label,
			phase=0,
			loss_weight=0.0625,
			loss_param=dict(ignore_label=ignore_label))

	### loss 2
	net.score2 = L.Convolution(net.u2d_conv,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore2 = L.Deconvolution(net.score2,
		param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=num_class,  pad=2, kernel_size=8, stride=4,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss2 = L.SoftmaxWithLoss(net.upscore2, net.label,
			phase=0,
			loss_weight=0.125,
			loss_param=dict(ignore_label=ignore_label))

	### loss 1
	net.score1 = L.Convolution(net.u1d_conv,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore1 = L.Deconvolution(net.score1,
		param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=num_class,  pad=1, kernel_size=4, stride=2,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss1 = L.SoftmaxWithLoss(net.upscore1, net.label,
			phase=0,
			loss_weight=0.25,
			loss_param=dict(ignore_label=ignore_label))
	### mask1
	net.prob = L.Softmax(net.upscore1)
	net.slice10, net.slice11, net.slice12 = L.Slice(net.prob, slice_param=dict(axis=1, slice_point=[1,2]), ntop=3)
	net.mask1 = L.Eltwise(net.slice11, net.slice12, eltwise_param=dict(operation=1))
	net.tile1 = L.Tile(net.mask1, axis=1, tiles=128)
	## u0 ############
	### a ### Fourth Deconvolution
	net.u0a_dconv, net.u0a_bn, net.u0a_scale, net.u0a_relu = deconv_bn_scale_relu(net.u1d_conv, 64, pad=0, kernel_size=2, stride=2, phase=phase)
	### b ### Crop and Concat
	#net.u0b_crop = L.Crop(net.u0a_dconv, net.d0c_conv, axis=2, offset=0)
	#net.u0b_crop = crop(net.u0a_relu, net.d0c_relu)
	net.u0b_concat = L.Concat(net.u0a_dconv, net.d0c_conv, axis=1)
	net.u0_eltw = L.Eltwise(net.u0b_concat, net.tile1, eltwise_param=dict(operation=0))
	### c ###
	net.u0c_conv, net.u0c_bn, net.u0c_scale, net.u0c_relu = conv_bn_scale_relu(net.u0_eltw, 64, pad=1, kernel_size=3, stride=1, phase=phase)
	### d ###
	net.u0d_conv, net.u0d_bn, net.u0d_scale, net.u0d_relu = conv_bn_scale_relu(net.u0c_conv, 64, pad=1, kernel_size=3, stride=1, phase=phase)


	### loss 0
	net.score = L.Convolution(net.u0d_conv,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	if phase == "train":
		net.loss = L.SoftmaxWithLoss(net.score, net.label,
			phase=0,
			loss_weight=1,
			loss_param=dict(ignore_label=ignore_label))

	return net.to_proto()


def unet_rcu_2d_bn(dim_data, dim_label, num_class, ignore_label=255, phase="train"):
	net = caffe.NetSpec()
	############ d0 ############
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if phase == "train":
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		# net.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))	
	net.d0b_conv, net.d0b_bn, net.d0b_scale, net.d0b_relu = conv_bn_scale_relu(net.data, 64, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d0c_conv, net.d0c_bn, net.d0c_scale, net.d0c_relu = conv_bn_scale_relu(net.d0b_conv, 64, pad=1, kernel_size=3, stride=1, phase=phase)
	############ d1 ############
	net.d1a_pool = max_pool(net.d0c_conv, pad=0, kernel_size=2, stride=2)
	net.d1b_conv, net.d1b_bn, net.d1b_scale, net.d1b_relu = conv_bn_scale_relu(net.d1a_pool, 128, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d1c_conv, net.d1c_bn, net.d1c_scale, net.d1c_relu = conv_bn_scale_relu(net.d1b_conv, 128, pad=1, kernel_size=3, stride=1, phase=phase)
	############ d2 ############
	net.d2a_pool = max_pool(net.d1c_conv, pad=0, kernel_size=2, stride=2)
	net.d2b_conv, net.d2b_bn, net.d2b_scale, net.d2b_relu = conv_bn_scale_relu(net.d2a_pool, 256, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d2c_conv, net.d2c_bn, net.d2c_scale, net.d2c_relu = conv_bn_scale_relu(net.d2b_conv, 256, pad=1, kernel_size=3, stride=1, phase=phase)
	############ d3 ############
	net.d3a_pool = max_pool(net.d2c_conv, pad=0, kernel_size=2, stride=2)
	net.d3b_conv, net.d3b_bn, net.d3b_scale, net.d3b_relu = conv_bn_scale_relu(net.d3a_pool, 512, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d3c_conv, net.d3c_bn, net.d3c_scale, net.d3c_relu = conv_bn_scale_relu(net.d3b_conv, 512, pad=1, kernel_size=3, stride=1, phase=phase)
	############ d4 ############
	net.d4a_pool = max_pool(net.d3c_conv, pad=0, kernel_size=2, stride=2)
	net.d4b_conv, net.d4b_bn, net.d4b_scale, net.d4b_relu = conv_bn_scale_relu(net.d4a_pool, 1024, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d4c_conv, net.d4c_bn, net.d4c_scale, net.d4c_relu = conv_bn_scale_relu(net.d4b_conv, 1024, pad=1, kernel_size=3, stride=1, phase=phase)

	############ u3 ############
	### a ### First Deconvolution
	net.u3a_dconv, net.u3a_bn, net.u3a_scale, net.u3a_relu = deconv_bn_scale_relu(net.d4c_conv, 512, pad=0, kernel_size=2, stride=2, phase=phase)
	### b ### Crop and Concat
	#net.u3b_crop = L.Crop(net.u3a_dconv, net.d3c_conv, axis=2, offset=0)
	#net.u3b_crop = crop(net.u3a_relu, net.d3c_relu)
	net.rcu3_conv1, net.rcu3_bn1, net.rcu3_scale1, net.rcu3_relu1, net.rcu3_conv2, net.rcu3_bn2, net.rcu3_scale2, net.rcu3_eltw, net.rcu3_relu2 = rcu_bn(net.d3c_conv, 512, pad=1, kernel_size=3, stride=1)
	net.u3b_concat = L.Concat(net.u3a_dconv, net.rcu3_eltw, axis=1)
	### c ###
	net.u3c_conv, net.u3c_bn, net.u3c_scale, net.u3c_relu = conv_bn_scale_relu(net.u3b_concat, 512, pad=1, kernel_size=3, stride=1, phase=phase)
	### d ###
	net.u3d_conv, net.u3d_bn, net.u3d_scale, net.u3d_relu = conv_bn_scale_relu(net.u3c_conv, 512, pad=1, kernel_size=3, stride=1, phase=phase)

	############ u2 ############
	### a ### Second Deconvolution
	net.u2a_dconv, net.u2a_bn, net.u2a_scale, net.u2a_relu = deconv_bn_scale_relu(net.u3d_conv, 256, pad=0, kernel_size=2, stride=2, phase=phase)
	### b ### Crop and Concat
	#net.u2b_crop = L.Crop(net.u2a_dconv, net.d2c_conv, axis=2, offset=0)
	#net.u2b_crop = crop(net.u2a_relu, net.d2c_relu)
	net.rcu2_conv1, net.rcu2_bn1, net.rcu2_scale1, net.rcu2_relu1, net.rcu2_conv2, net.rcu2_bn2, net.rcu2_scale2, net.rcu2_eltw, net.rcu2_relu2 = rcu_bn(net.d2c_conv, 256, pad=1, kernel_size=3, stride=1)
	net.u2b_concat = L.Concat(net.u2a_dconv, net.rcu2_eltw, axis=1)
	### c ###
	net.u2c_conv, net.u2c_bn, net.u2c_scale, net.u2c_relu = conv_bn_scale_relu(net.u2b_concat, 256, pad=1, kernel_size=3, stride=1, phase=phase)
	### d ###
	net.u2d_conv, net.u2d_bn, net.u2d_scale, net.u2d_relu = conv_bn_scale_relu(net.u2c_conv, 256, pad=1, kernel_size=3, stride=1, phase=phase)

	############ u1 ############
	### a ### Third Deconvolution
	net.u1a_dconv, net.u1a_bn, net.u1a_scale, net.u1a_relu = deconv_bn_scale_relu(net.u2d_conv, 128, pad=0, kernel_size=2, stride=2, phase=phase)
	### b ### Crop and Concat
	#net.u1b_crop = L.Crop(net.u1a_dconv, net.d1c_conv, axis=2, offset=0)
	#net.u1b_crop = crop(net.u1a_relu, net.d1c_relu)
	net.rcu1_conv1, net.rcu1_bn1, net.rcu1_scale1, net.rcu1_relu1, net.rcu1_conv2, net.rcu1_bn2, net.rcu1_scale2, net.rcu1_eltw, net.rcu1_relu2 = rcu_bn(net.d1c_conv, 128, pad=1, kernel_size=3, stride=1)
	net.u1b_concat = L.Concat(net.u1a_dconv, net.rcu1_eltw, axis=1)
	### c ###
	net.u1c_conv, net.u1c_bn, net.u1c_scale, net.u1c_relu = conv_bn_scale_relu(net.u1b_concat, 128, pad=1, kernel_size=3, stride=1, phase=phase)
	### d ###
	net.u1d_conv, net.u1d_bn, net.u1d_scale, net.u1d_relu = conv_bn_scale_relu(net.u1c_conv, 128, pad=1, kernel_size=3, stride=1, phase=phase)

	############ u0 ############
	### a ### Fourth Deconvolution
	net.u0a_dconv, net.u0a_bn, net.u0a_scale, net.u0a_relu = deconv_bn_scale_relu(net.u1d_conv, 64, pad=0, kernel_size=2, stride=2, phase=phase)
	### b ### Crop and Concat
	#net.u0b_crop = L.Crop(net.u0a_dconv, net.d0c_conv, axis=2, offset=0)
	#net.u0b_crop = crop(net.u0a_relu, net.d0c_relu)
	net.rcu0_conv1, net.rcu0_bn1, net.rcu0_scale1, net.rcu0_relu1, net.rcu0_conv2, net.rcu0_bn2, net.rcu0_scale2, net.rcu0_eltw, net.rcu0_relu2 = rcu_bn(net.d0c_conv, 64, pad=1, kernel_size=3, stride=1)
	net.u0b_concat = L.Concat(net.u0a_dconv, net.rcu0_eltw, axis=1)
	### c ###
	net.u0c_conv, net.u0c_bn, net.u0c_scale, net.u0c_relu = conv_bn_scale_relu(net.u0b_concat, 64, pad=1, kernel_size=3, stride=1, phase=phase)
	### d ###
	net.u0d_conv, net.u0d_bn, net.u0d_scale, net.u0d_relu = conv_bn_scale_relu(net.u0c_conv, 64, pad=1, kernel_size=3, stride=1, phase=phase)

	# ############ Score ############
	# net.score = L.Convolution(net.u0d_conv,
	# 	param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
	# 	num_output=num_class, pad=0, kernel_size=1, stride=1, 
	# 	# weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
	# 	weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
	# 	engine=engine)

	# ############ Loss ############
	# if phase == "train":
	# 	net.loss = L.SoftmaxWithLoss(net.score, net.label,
	# 		phase=0,
	# 		loss_weight=1,
	# 		loss_param=dict(ignore_label=ignore_label))
	# 	# net.loss = L.WeightedSoftmaxWithLoss(net.score, net.label, net.label_weight,
	# 	# 	phase=0,
	# 	# 	loss_weight=1,
	# 	# 	loss_param=dict(ignore_label=ignore_label))

	# ### loss 4 
	# net.score4 = L.Convolution(net.d4c_conv,
	# 	param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
	# 	num_output=num_class, pad=0, kernel_size=1, stride=1, 
	# 	weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	# net.upscore4 = L.Deconvolution(net.score4,
	# 	param=[dict(lr_mult=1, decay_mult=1)],
	# 	convolution_param=dict(num_output=num_class,  pad=8, kernel_size=32, stride=16,
	# 		weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	# if phase == "train":
	# 	net.loss4 = L.SoftmaxWithLoss(net.upscore4, net.label,
	# 		phase=0,
	# 		loss_weight=0.0625,
	# 		loss_param=dict(ignore_label=ignore_label))

	### loss 3
	net.score3 = L.Convolution(net.u3d_conv,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore3 = L.Deconvolution(net.score3,
		param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=num_class,  pad=4, kernel_size=16, stride=8,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss3 = L.SoftmaxWithLoss(net.upscore3, net.label,
			phase=0,
			loss_weight=0.0625,
			loss_param=dict(ignore_label=ignore_label))

	### loss 2
	net.score2 = L.Convolution(net.u2d_conv,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore2 = L.Deconvolution(net.score2,
		param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=num_class,  pad=2, kernel_size=8, stride=4,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss2 = L.SoftmaxWithLoss(net.upscore2, net.label,
			phase=0,
			loss_weight=0.125,
			loss_param=dict(ignore_label=ignore_label))

	### loss 1
	net.score1 = L.Convolution(net.u1d_conv,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore1 = L.Deconvolution(net.score1,
		param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=num_class,  pad=1, kernel_size=4, stride=2,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss1 = L.SoftmaxWithLoss(net.upscore1, net.label,
			phase=0,
			loss_weight=0.25,
			loss_param=dict(ignore_label=ignore_label))

	### loss 0
	net.score = L.Convolution(net.u0d_conv,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	if phase == "train":
		net.loss = L.SoftmaxWithLoss(net.score, net.label,
			phase=0,
			loss_weight=0.5,
			loss_param=dict(ignore_label=ignore_label))

	return net.to_proto()

def unet_2d_bn_8s(dim_data, dim_label, num_class, ignore_label=255, phase="train"):
	net = caffe.NetSpec()
	############ d0 ############
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if phase == "train":
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		# net.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))	
	net.d0b_conv, net.d0b_bn, net.d0b_scale, net.d0b_relu = conv_bn_scale_relu(net.data, 64, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d0c_conv, net.d0c_bn, net.d0c_scale, net.d0c_relu = conv_bn_scale_relu(net.d0b_conv, 64, pad=1, kernel_size=3, stride=1, phase=phase)
	############ d1 ############
	net.d1a_pool = max_pool(net.d0c_conv, pad=0, kernel_size=2, stride=2)
	net.d1b_conv, net.d1b_bn, net.d1b_scale, net.d1b_relu = conv_bn_scale_relu(net.d1a_pool, 128, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d1c_conv, net.d1c_bn, net.d1c_scale, net.d1c_relu = conv_bn_scale_relu(net.d1b_conv, 128, pad=1, kernel_size=3, stride=1, phase=phase)
	############ d2 ############
	net.d2a_pool = max_pool(net.d1c_conv, pad=0, kernel_size=2, stride=2)
	net.d2b_conv, net.d2b_bn, net.d2b_scale, net.d2b_relu = conv_bn_scale_relu(net.d2a_pool, 256, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d2c_conv, net.d2c_bn, net.d2c_scale, net.d2c_relu = conv_bn_scale_relu(net.d2b_conv, 256, pad=1, kernel_size=3, stride=1, phase=phase)
	############ d3 ############
	net.d3a_pool = max_pool(net.d2c_conv, pad=0, kernel_size=2, stride=2)
	net.d3b_conv, net.d3b_bn, net.d3b_scale, net.d3b_relu = conv_bn_scale_relu(net.d3a_pool, 512, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d3c_conv, net.d3c_bn, net.d3c_scale, net.d3c_relu = conv_bn_scale_relu(net.d3b_conv, 512, pad=1, kernel_size=3, stride=1, phase=phase)
	############ d4 ############
	# net.d4a_pool = max_pool(net.d3c_conv, pad=0, kernel_size=2, stride=2)
	# net.d4a_conv, net.d4a_bn, net.d4a_scale, net.d4a_relu = dila_conv_bn_scale_relu(net.d3c_conv, 1024, pad=2, kernel_size=3, stride=1, dilation=2, phase=phase)
	# net.d4b_conv, net.d4b_bn, net.d4b_scale, net.d4b_relu = conv_bn_scale_relu(net.d4a_conv, 1024, pad=1, kernel_size=3, stride=1, phase=phase)
	# net.d4c_conv, net.d4c_bn, net.d4c_scale, net.d4c_relu = conv_bn_scale_relu(net.d4b_conv, 1024, pad=1, kernel_size=3, stride=1, phase=phase)

	########### u3 ############
	### a ### First Deconvolution
	# net.u3a_dconv, net.u3a_bn, net.u3a_scale, net.u3a_relu = deconv_bn_scale_relu(net.d4c_conv, 512, pad=0, kernel_size=2, stride=2, phase=phase)
	### b ### Crop and Concat
	#net.u3b_crop = L.Crop(net.u3a_dconv, net.d3c_conv, axis=2, offset=0)
	# #net.u3b_crop = crop(net.u3a_relu, net.d3c_relu)
	# net.u3b_concat = L.Concat(net.d4c_conv, net.d3c_conv, axis=1)
	# ### c ###
	# net.u3c_conv, net.u3c_bn, net.u3c_scale, net.u3c_relu = conv_bn_scale_relu(net.u3b_concat, 512, pad=1, kernel_size=3, stride=1, phase=phase)
	# ### d ###
	# net.u3d_conv, net.u3d_bn, net.u3d_scale, net.u3d_relu = conv_bn_scale_relu(net.u3c_conv, 512, pad=1, kernel_size=3, stride=1, phase=phase)

	############ u2 ############
	### a ### Second Deconvolution
	net.u2a_dconv, net.u2a_bn, net.u2a_scale, net.u2a_relu = deconv_bn_scale_relu(net.d3c_conv, 256, pad=0, kernel_size=2, stride=2, phase=phase)
	### b ### Crop and Concat
	#net.u2b_crop = L.Crop(net.u2a_dconv, net.d2c_conv, axis=2, offset=0)
	#net.u2b_crop = crop(net.u2a_relu, net.d2c_relu)
	net.u2b_concat = L.Concat(net.u2a_dconv, net.d2c_conv, axis=1)
	### c ###
	net.u2c_conv, net.u2c_bn, net.u2c_scale, net.u2c_relu = conv_bn_scale_relu(net.u2b_concat, 256, pad=1, kernel_size=3, stride=1, phase=phase)
	### d ###
	net.u2d_conv, net.u2d_bn, net.u2d_scale, net.u2d_relu = conv_bn_scale_relu(net.u2c_conv, 256, pad=1, kernel_size=3, stride=1, phase=phase)

	############ u1 ############
	### a ### Third Deconvolution
	net.u1a_dconv, net.u1a_bn, net.u1a_scale, net.u1a_relu = deconv_bn_scale_relu(net.u2d_conv, 128, pad=0, kernel_size=2, stride=2, phase=phase)
	### b ### Crop and Concat
	#net.u1b_crop = L.Crop(net.u1a_dconv, net.d1c_conv, axis=2, offset=0)
	#net.u1b_crop = crop(net.u1a_relu, net.d1c_relu)
	net.u1b_concat = L.Concat(net.u1a_dconv, net.d1c_conv, axis=1)
	### c ###
	net.u1c_conv, net.u1c_bn, net.u1c_scale, net.u1c_relu = conv_bn_scale_relu(net.u1b_concat, 128, pad=1, kernel_size=3, stride=1, phase=phase)
	### d ###
	net.u1d_conv, net.u1d_bn, net.u1d_scale, net.u1d_relu = conv_bn_scale_relu(net.u1c_conv, 128, pad=1, kernel_size=3, stride=1, phase=phase)

	############ u0 ############
	### a ### Fourth Deconvolution
	net.u0a_dconv, net.u0a_bn, net.u0a_scale, net.u0a_relu = deconv_bn_scale_relu(net.u1d_conv, 64, pad=0, kernel_size=2, stride=2, phase=phase)
	### b ### Crop and Concat
	#net.u0b_crop = L.Crop(net.u0a_dconv, net.d0c_conv, axis=2, offset=0)
	#net.u0b_crop = crop(net.u0a_relu, net.d0c_relu)
	net.u0b_concat = L.Concat(net.u0a_dconv, net.d0c_conv, axis=1)
	### c ###
	net.u0c_conv, net.u0c_bn, net.u0c_scale, net.u0c_relu = conv_bn_scale_relu(net.u0b_concat, 64, pad=1, kernel_size=3, stride=1, phase=phase)
	### d ###
	net.u0d_conv, net.u0d_bn, net.u0d_scale, net.u0d_relu = conv_bn_scale_relu(net.u0c_conv, 64, pad=1, kernel_size=3, stride=1, phase=phase)

	# ############ Score ############
	# net.score = L.Convolution(net.u0d_conv,
	# 	param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
	# 	num_output=num_class, pad=0, kernel_size=1, stride=1, 
	# 	# weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
	# 	weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
	# 	engine=engine)

	# ############ Loss ############
	# if phase == "train":
	# 	net.loss = L.SoftmaxWithLoss(net.score, net.label,
	# 		phase=0,
	# 		loss_weight=1,
	# 		loss_param=dict(ignore_label=ignore_label))
	# 	# net.loss = L.WeightedSoftmaxWithLoss(net.score, net.label, net.label_weight,
	# 	# 	phase=0,
	# 	# 	loss_weight=1,
	# 	# 	loss_param=dict(ignore_label=ignore_label))

	# ### loss 4 
	# net.score4 = L.Convolution(net.d4c_conv,
	# 	param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
	# 	num_output=num_class, pad=0, kernel_size=1, stride=1, 
	# 	weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	# net.upscore4 = L.Deconvolution(net.score4,
	# 	param=[dict(lr_mult=1, decay_mult=1)],
	# 	convolution_param=dict(num_output=num_class,  pad=8, kernel_size=32, stride=16,
	# 		weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	# if phase == "train":
	# 	net.loss4 = L.SoftmaxWithLoss(net.upscore4, net.label,
	# 		phase=0,
	# 		loss_weight=0.0625,
	# 		loss_param=dict(ignore_label=ignore_label))

	### loss 3
	# net.score3 = L.Convolution(net.u3d_conv,
	# 	param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
	# 	num_output=num_class, pad=0, kernel_size=1, stride=1, 
	# 	weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	# net.upscore3 = L.Deconvolution(net.score3,
	# 	param=[dict(lr_mult=1, decay_mult=1)],
	# 	convolution_param=dict(num_output=num_class,  pad=4, kernel_size=16, stride=8,
	# 		weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	# if phase == "train":
	# 	net.loss3 = L.SoftmaxWithLoss(net.upscore3, net.label,
	# 		phase=0,
	# 		loss_weight=0.0625,
	# 		loss_param=dict(ignore_label=ignore_label))

	### loss 2
	net.score2 = L.Convolution(net.u2d_conv,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore2 = L.Deconvolution(net.score2,
		param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=num_class,  pad=2, kernel_size=8, stride=4,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss2 = L.SoftmaxWithLoss(net.upscore2, net.label,
			phase=0,
			loss_weight=0.25,
			loss_param=dict(ignore_label=ignore_label))

	### loss 1
	net.score1 = L.Convolution(net.u1d_conv,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore1 = L.Deconvolution(net.score1,
		param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=num_class,  pad=1, kernel_size=4, stride=2,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss1 = L.SoftmaxWithLoss(net.upscore1, net.label,
			phase=0,
			loss_weight=0.5,
			loss_param=dict(ignore_label=ignore_label))

	### loss 0
	net.score = L.Convolution(net.u0d_conv,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	if phase == "train":
		net.loss = L.SoftmaxWithLoss(net.score, net.label,
			phase=0,
			loss_weight=1,
			loss_param=dict(ignore_label=ignore_label))
	else:
		net.prob = L.Softmax(net.score, axis=1)

	return net.to_proto()

############ ############ ############ ############ ############ ############
def unet_2rcu_2d(dim_data, dim_label, num_class, ignore_label=255, phase="train"):
	net = caffe.NetSpec()
	############ d0 ############
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if phase == "train":
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		# net.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))
	net.d0b_conv, net.d0b_relu = conv_relu(net.data, 64, pad=1, kernel_size=3, stride=1)
	net.d0c_conv, net.d0c_relu = conv_relu(net.d0b_conv, 64, pad=1, kernel_size=3, stride=1)
	############ d1 ############
	net.d1a_pool = max_pool(net.d0c_conv, pad=0, kernel_size=2, stride=2)
	net.d1b_conv, net.d1b_relu = conv_relu(net.d1a_pool, 128, pad=1, kernel_size=3, stride=1)
	net.d1c_conv, net.d1c_relu = conv_relu(net.d1b_conv, 128, pad=1, kernel_size=3, stride=1)
	############ d2 ############
	net.d2a_pool = max_pool(net.d1c_conv, pad=0, kernel_size=2, stride=2)
	net.d2b_conv, net.d2b_relu = conv_relu(net.d2a_pool, 256, pad=1, kernel_size=3, stride=1)
	net.d2c_conv, net.d2c_relu = conv_relu(net.d2b_conv, 256, pad=1, kernel_size=3, stride=1)
	############ d3 ############
	net.d3a_pool = max_pool(net.d2c_conv, pad=0, kernel_size=2, stride=2)
	net.d3b_conv, net.d3b_relu = conv_relu(net.d3a_pool, 512, pad=1, kernel_size=3, stride=1)
	net.d3c_conv, net.d3c_relu = conv_relu(net.d3b_conv, 512, pad=1, kernel_size=3, stride=1)
	############ d4 ############
	net.d4a_pool = max_pool(net.d3c_conv, pad=0, kernel_size=2, stride=2)
	net.d4b_conv, net.d4b_relu = conv_relu(net.d4a_pool, 1024, pad=1, kernel_size=3, stride=1)
	net.d4c_conv, net.d4c_relu = conv_relu(net.d4b_conv, 1024, pad=1, kernel_size=3, stride=1)

	############ u3 ############
	### a ### First Deconvolution
	net.u3a_dconv, net.u3a_relu = deconv_relu(net.d4c_conv, 512, pad=0, kernel_size=2, stride=2)
	### b ### Crop and Concat
	#net.u3b_crop = L.Crop(net.u3a_relu, net.d3c_relu, axis=2, offset=0)
	### rcu3
	net.rcu3a_conv1, net.rcu3a_relu1, net.rcu3a_conv2, net.rcu3a_eltw, net.rcu3a_relu2 = rcu(net.d3c_conv, 512, pad=1, kernel_size=3, stride=1)
	net.rcu3b_conv1, net.rcu3b_relu1, net.rcu3b_conv2, net.rcu3b_eltw, net.rcu3b_relu2 = rcu(net.rcu3a_eltw, 512, pad=1, kernel_size=3, stride=1)
	net.u3b_concat = L.Concat(net.u3a_dconv, net.rcu3b_eltw, axis=1)
	### c ###
	net.u3c_conv, net.u3c_relu = conv_relu(net.u3b_concat, 512, pad=1, kernel_size=3, stride=1)
	### d ###
	net.u3d_conv, net.u3d_relu = conv_relu(net.u3c_conv, 512, pad=1, kernel_size=3, stride=1)

	############ u2 ############
	### a ### Second Deconvolution
	net.u2a_dconv, net.u2a_relu = deconv_relu(net.u3d_conv, 256, pad=0, kernel_size=2, stride=2)
	### b ### Crop and Concat
	#net.u2b_crop = L.Crop(net.u2a_relu, net.d2c_relu, axis=2, offset=0)
	### rcu2
	net.rcu2a_conv1, net.rcu2a_relu1, net.rcu2a_conv2, net.rcu2a_eltw, net.rcu2a_relu2 = rcu(net.d2c_conv, 256, pad=1, kernel_size=3, stride=1)
	net.rcu2b_conv1, net.rcu2b_relu1, net.rcu2b_conv2, net.rcu2b_eltw, net.rcu2b_relu2 = rcu(net.rcu2a_eltw, 256, pad=1, kernel_size=3, stride=1)
	net.u2b_concat = L.Concat(net.u2a_dconv, net.rcu2b_eltw, axis=1)
	### c ###
	net.u2c_conv, net.u2c_relu = conv_relu(net.u2b_concat, 256, pad=1, kernel_size=3, stride=1)
	### d ###
	net.u2d_conv, net.u2d_relu = conv_relu(net.u2c_conv, 256, pad=1, kernel_size=3, stride=1)

	############ u1 ############
	### a ### Third Deconvolution
	net.u1a_dconv, net.u1a_relu = deconv_relu(net.u2d_conv, 128, pad=0, kernel_size=2, stride=2)
	### b ### Crop and Concat
	#net.u1b_crop = L.Crop(net.u1a_relu, net.d1c_relu, axis=2, offset=0)
	### rcu1
	net.rcu1a_conv1, net.rcu1a_relu1, net.rcu1a_conv2, net.rcu1a_eltw, net.rcu1a_relu2 = rcu(net.d1c_conv, 128, pad=1, kernel_size=3, stride=1)
	net.rcu1b_conv1, net.rcu1b_relu1, net.rcu1b_conv2, net.rcu1b_eltw, net.rcu1b_relu2 = rcu(net.rcu1a_eltw, 128, pad=1, kernel_size=3, stride=1)
	net.u1b_concat = L.Concat(net.u1a_dconv, net.rcu1b_eltw, axis=1)
	### c ###
	net.u1c_conv, net.u1c_relu = conv_relu(net.u1b_concat, 128, pad=1, kernel_size=3, stride=1)
	### d ###
	net.u1d_conv, net.u1d_relu = conv_relu(net.u1c_conv, 128, pad=1, kernel_size=3, stride=1)

	############ u0 ############
	### a ### Fourth Deconvolution
	net.u0a_dconv, net.u0a_relu = deconv_relu(net.u1d_conv, 64, pad=0, kernel_size=2, stride=2)
	### b ### Crop and Concat
	#net.u0b_crop = L.Crop(net.u0a_relu, net.d0c_relu, axis=2, offset=0)
	### rcu0
	net.rcu0a_conv1, net.rcu0a_relu1, net.rcu0a_conv2, net.rcu0a_eltw, net.rcu0a_relu2 = rcu(net.d0c_conv, 64, pad=1, kernel_size=3, stride=1)
	net.rcu0b_conv1, net.rcu0b_relu1, net.rcu0b_conv2, net.rcu0b_eltw, net.rcu0b_relu2 = rcu(net.rcu0a_eltw, 64, pad=1, kernel_size=3, stride=1)
	net.u0b_concat = L.Concat(net.u0a_dconv, net.rcu0b_eltw, axis=1)
	### c ###
	net.u0c_conv, net.u0c_relu = conv_relu(net.u0b_concat, 64, pad=1, kernel_size=3, stride=1)
	### d ###
	net.u0d_conv, net.u0d_relu = conv_relu(net.u0c_conv, 64, pad=1, kernel_size=3, stride=1)

	############ Score ############
	net.score = L.Convolution(net.u0d_conv,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		#weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=engine)

	############ Loss ############
	if phase == "train":
		net.loss = L.SoftmaxWithLoss(net.score, net.label,
			phase=0,
			loss_weight=1,
			loss_param=dict(ignore_label=ignore_label))
		# net.loss = L.WeightedSoftmaxWithLoss(net.score, net.label, net.label_weight,
		# 	phase=0,
		# 	loss_weight=1,
		# 	loss_param=dict(ignore_label=ignore_label))

	return net.to_proto()

def unet_rcu_2d(dim_data, dim_label, num_class, ignore_label=255, phase="train"):
	net = caffe.NetSpec()
	############ d0 ############
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if phase == "train":
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		# net.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))
	net.d0b_conv, net.d0b_relu = conv_relu(net.data, 64, pad=1, kernel_size=3, stride=1)
	net.d0c_conv, net.d0c_relu = conv_relu(net.d0b_conv, 64, pad=1, kernel_size=3, stride=1)
	############ d1 ############
	net.d1a_pool = max_pool(net.d0c_conv, pad=0, kernel_size=2, stride=2)
	net.d1b_conv, net.d1b_relu = conv_relu(net.d1a_pool, 128, pad=1, kernel_size=3, stride=1)
	net.d1c_conv, net.d1c_relu = conv_relu(net.d1b_conv, 128, pad=1, kernel_size=3, stride=1)
	############ d2 ############
	net.d2a_pool = max_pool(net.d1c_conv, pad=0, kernel_size=2, stride=2)
	net.d2b_conv, net.d2b_relu = conv_relu(net.d2a_pool, 256, pad=1, kernel_size=3, stride=1)
	net.d2c_conv, net.d2c_relu = conv_relu(net.d2b_conv, 256, pad=1, kernel_size=3, stride=1)
	############ d3 ############
	net.d3a_pool = max_pool(net.d2c_conv, pad=0, kernel_size=2, stride=2)
	net.d3b_conv, net.d3b_relu = conv_relu(net.d3a_pool, 512, pad=1, kernel_size=3, stride=1)
	net.d3c_conv, net.d3c_relu = conv_relu(net.d3b_conv, 512, pad=1, kernel_size=3, stride=1)
	############ d4 ############
	net.d4a_pool = max_pool(net.d3c_conv, pad=0, kernel_size=2, stride=2)
	net.d4b_conv, net.d4b_relu = conv_relu(net.d4a_pool, 1024, pad=1, kernel_size=3, stride=1)
	net.d4c_conv, net.d4c_relu = conv_relu(net.d4b_conv, 1024, pad=1, kernel_size=3, stride=1)

	############ u3 ############
	### a ### First Deconvolution
	net.u3a_dconv, net.u3a_relu = deconv_relu(net.d4c_conv, 512, pad=0, kernel_size=2, stride=2)
	### b ### Crop and Concat
	#net.u3b_crop = L.Crop(net.u3a_relu, net.d3c_relu, axis=2, offset=0)
	### rcu3
	net.rcu3_conv1, net.rcu3_relu1, net.rcu3_conv2, net.rcu3_eltw, net.rcu3_relu2 = rcu(net.d3c_conv, 512, pad=1, kernel_size=3, stride=1)
	net.u3b_concat = L.Concat(net.u3a_dconv, net.rcu3_eltw, axis=1)
	### c ###
	net.u3c_conv, net.u3c_relu = conv_relu(net.u3b_concat, 512, pad=1, kernel_size=3, stride=1)
	### d ###
	net.u3d_conv, net.u3d_relu = conv_relu(net.u3c_conv, 512, pad=1, kernel_size=3, stride=1)

	############ u2 ############
	### a ### Second Deconvolution
	net.u2a_dconv, net.u2a_relu = deconv_relu(net.u3d_conv, 256, pad=0, kernel_size=2, stride=2)
	### b ### Crop and Concat
	#net.u2b_crop = L.Crop(net.u2a_relu, net.d2c_relu, axis=2, offset=0)
	### rcu2
	net.rcu2_conv1, net.rcu2_relu1, net.rcu2_conv2, net.rcu2_eltw, net.rcu2_relu2 = rcu(net.d2c_conv, 256, pad=1, kernel_size=3, stride=1)
	net.u2b_concat = L.Concat(net.u2a_dconv, net.rcu2_eltw, axis=1)
	### c ###
	net.u2c_conv, net.u2c_relu = conv_relu(net.u2b_concat, 256, pad=1, kernel_size=3, stride=1)
	### d ###
	net.u2d_conv, net.u2d_relu = conv_relu(net.u2c_conv, 256, pad=1, kernel_size=3, stride=1)

	############ u1 ############
	### a ### Third Deconvolution
	net.u1a_dconv, net.u1a_relu = deconv_relu(net.u2d_conv, 128, pad=0, kernel_size=2, stride=2)
	### b ### Crop and Concat
	#net.u1b_crop = L.Crop(net.u1a_relu, net.d1c_relu, axis=2, offset=0)
	### rcu1
	net.rcu1_conv1, net.rcu1_relu1, net.rcu1_conv2, net.rcu1_eltw, net.rcu1_relu2 = rcu(net.d1c_conv, 128, pad=1, kernel_size=3, stride=1)
	net.u1b_concat = L.Concat(net.u1a_dconv, net.rcu1_eltw, axis=1)
	### c ###
	net.u1c_conv, net.u1c_relu = conv_relu(net.u1b_concat, 128, pad=1, kernel_size=3, stride=1)
	### d ###
	net.u1d_conv, net.u1d_relu = conv_relu(net.u1c_conv, 128, pad=1, kernel_size=3, stride=1)

	############ u0 ############
	### a ### Fourth Deconvolution
	net.u0a_dconv, net.u0a_relu = deconv_relu(net.u1d_conv, 64, pad=0, kernel_size=2, stride=2)
	### b ### Crop and Concat
	#net.u0b_crop = L.Crop(net.u0a_relu, net.d0c_relu, axis=2, offset=0)
	### rcu0
	net.rcu0_conv1, net.rcu0_relu1, net.rcu0_conv2, net.rcu0_eltw, net.rcu0_relu2 = rcu(net.d0c_conv, 64, pad=1, kernel_size=3, stride=1)
	net.u0b_concat = L.Concat(net.u0a_dconv, net.rcu0_eltw, axis=1)
	### c ###
	net.u0c_conv, net.u0c_relu = conv_relu(net.u0b_concat, 64, pad=1, kernel_size=3, stride=1)
	### d ###
	net.u0d_conv, net.u0d_relu = conv_relu(net.u0c_conv, 64, pad=1, kernel_size=3, stride=1)

	############ Score ############
	net.score = L.Convolution(net.u0d_conv,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		#weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=engine)

	############ Loss ############
	if phase == "train":
		net.loss = L.SoftmaxWithLoss(net.score, net.label,
			phase=0,
			loss_weight=1,
			loss_param=dict(ignore_label=ignore_label))
		# net.loss = L.WeightedSoftmaxWithLoss(net.score, net.label, net.label_weight,
		# 	phase=0,
		# 	loss_weight=1,
		# 	loss_param=dict(ignore_label=ignore_label))

	return net.to_proto()


def unet_2d(dim_data, dim_label, num_class, ignore_label=255, phase="train"):
	net = caffe.NetSpec()
	############ d0 ############
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if phase == "train":
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		# net.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))
	net.d0b_conv, net.d0b_relu = conv_relu(net.data, 64, pad=1, kernel_size=3, stride=1)
	net.d0c_conv, net.d0c_relu = conv_relu(net.d0b_conv, 64, pad=1, kernel_size=3, stride=1)
	############ d1 ############
	net.d1a_pool = max_pool(net.d0c_conv, pad=0, kernel_size=2, stride=2)
	net.d1b_conv, net.d1b_relu = conv_relu(net.d1a_pool, 128, pad=1, kernel_size=3, stride=1)
	net.d1c_conv, net.d1c_relu = conv_relu(net.d1b_conv, 128, pad=1, kernel_size=3, stride=1)
	############ d2 ############
	net.d2a_pool = max_pool(net.d1c_conv, pad=0, kernel_size=2, stride=2)
	net.d2b_conv, net.d2b_relu = conv_relu(net.d2a_pool, 256, pad=1, kernel_size=3, stride=1)
	net.d2c_conv, net.d2c_relu = conv_relu(net.d2b_conv, 256, pad=1, kernel_size=3, stride=1)
	############ d3 ############
	net.d3a_pool = max_pool(net.d2c_conv, pad=0, kernel_size=2, stride=2)
	net.d3b_conv, net.d3b_relu = conv_relu(net.d3a_pool, 512, pad=1, kernel_size=3, stride=1)
	net.d3c_conv, net.d3c_relu = conv_relu(net.d3b_conv, 512, pad=1, kernel_size=3, stride=1)
	############ d4 ############
	net.d4a_pool = max_pool(net.d3c_conv, pad=0, kernel_size=2, stride=2)
	net.d4b_conv, net.d4b_relu = conv_relu(net.d4a_pool, 1024, pad=1, kernel_size=3, stride=1)
	net.d4c_conv, net.d4c_relu = conv_relu(net.d4b_conv, 1024, pad=1, kernel_size=3, stride=1)

	############ u3 ############
	### a ### First Deconvolution
	net.u3a_dconv, net.u3a_relu = deconv_relu(net.d4c_conv, 512, pad=0, kernel_size=2, stride=2)
	### b ### Crop and Concat
	#net.u3b_crop = L.Crop(net.u3a_relu, net.d3c_relu, axis=2, offset=0)
	net.u3b_concat = L.Concat(net.u3a_dconv, net.d3c_conv, axis=1)
	### c ###
	net.u3c_conv, net.u3c_relu = conv_relu(net.u3b_concat, 512, pad=1, kernel_size=3, stride=1)
	### d ###
	net.u3d_conv, net.u3d_relu = conv_relu(net.u3c_conv, 512, pad=1, kernel_size=3, stride=1)

	############ u2 ############
	### a ### Second Deconvolution
	net.u2a_dconv, net.u2a_relu = deconv_relu(net.u3d_conv, 256, pad=0, kernel_size=2, stride=2)
	### b ### Crop and Concat
	#net.u2b_crop = L.Crop(net.u2a_relu, net.d2c_relu, axis=2, offset=0)
	net.u2b_concat = L.Concat(net.u2a_dconv, net.d2c_conv, axis=1)
	### c ###
	net.u2c_conv, net.u2c_relu = conv_relu(net.u2b_concat, 256, pad=1, kernel_size=3, stride=1)
	### d ###
	net.u2d_conv, net.u2d_relu = conv_relu(net.u2c_conv, 256, pad=1, kernel_size=3, stride=1)

	############ u1 ############
	### a ### Third Deconvolution
	net.u1a_dconv, net.u1a_relu = deconv_relu(net.u2d_conv, 128, pad=0, kernel_size=2, stride=2)
	### b ### Crop and Concat
	#net.u1b_crop = L.Crop(net.u1a_relu, net.d1c_relu, axis=2, offset=0)
	net.u1b_concat = L.Concat(net.u1a_dconv, net.d1c_conv, axis=1)
	### c ###
	net.u1c_conv, net.u1c_relu = conv_relu(net.u1b_concat, 128, pad=1, kernel_size=3, stride=1)
	### d ###
	net.u1d_conv, net.u1d_relu = conv_relu(net.u1c_conv, 128, pad=1, kernel_size=3, stride=1)

	############ u0 ############
	### a ### Fourth Deconvolution
	net.u0a_dconv, net.u0a_relu = deconv_relu(net.u1d_conv, 64, pad=0, kernel_size=2, stride=2)
	### b ### Crop and Concat
	#net.u0b_crop = L.Crop(net.u0a_relu, net.d0c_relu, axis=2, offset=0)
	net.u0b_concat = L.Concat(net.u0a_dconv, net.d0c_conv, axis=1)
	### c ###
	net.u0c_conv, net.u0c_relu = conv_relu(net.u0b_concat, 64, pad=1, kernel_size=3, stride=1)
	### d ###
	net.u0d_conv, net.u0d_relu = conv_relu(net.u0c_conv, 64, pad=1, kernel_size=3, stride=1)

	############ Score ############
	net.score = L.Convolution(net.u0d_conv,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		#weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=engine)

	############ Loss ############
	if phase == "train":
		net.loss = L.SoftmaxWithLoss(net.score, net.label,
			phase=0,
			loss_weight=1,
			loss_param=dict(ignore_label=ignore_label))
		# net.loss = L.WeightedSoftmaxWithLoss(net.score, net.label, net.label_weight,
		# 	phase=0,
		# 	loss_weight=1,
		# 	loss_param=dict(ignore_label=ignore_label))

	return net.to_proto()

def unet_2d_bn(dim_data, dim_label, num_class, ignore_label=255, phase="train"):
	net = caffe.NetSpec()
	############ d0 ############
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if phase == "train":
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		# net.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))
	net.d0b_conv, net.d0b_bn, net.d0b_scale, net.d0b_relu = conv_bn_scale_relu(net.data, 64, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d0c_conv, net.d0c_bn, net.d0c_scale, net.d0c_relu = conv_bn_scale_relu(net.d0b_conv, 64, pad=1, kernel_size=3, stride=1, phase=phase)
	############ d1 ############
	net.d1a_pool = max_pool(net.d0c_conv, pad=0, kernel_size=2, stride=2)
	net.d1b_conv, net.d1b_bn, net.d1b_scale, net.d1b_relu = conv_bn_scale_relu(net.d1a_pool, 128, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d1c_conv, net.d1c_bn, net.d1c_scale, net.d1c_relu = conv_bn_scale_relu(net.d1b_conv, 128, pad=1, kernel_size=3, stride=1, phase=phase)
	############ d2 ############
	net.d2a_pool = max_pool(net.d1c_conv, pad=0, kernel_size=2, stride=2)
	net.d2b_conv, net.d2b_bn, net.d2b_scale, net.d2b_relu = conv_bn_scale_relu(net.d2a_pool, 256, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d2c_conv, net.d2c_bn, net.d2c_scale, net.d2c_relu = conv_bn_scale_relu(net.d2b_conv, 256, pad=1, kernel_size=3, stride=1, phase=phase)
	############ d3 ############
	net.d3a_pool = max_pool(net.d2c_conv, pad=0, kernel_size=2, stride=2)
	net.d3b_conv, net.d3b_bn, net.d3b_scale, net.d3b_relu = conv_bn_scale_relu(net.d3a_pool, 512, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d3c_conv, net.d3c_bn, net.d3c_scale, net.d3c_relu = conv_bn_scale_relu(net.d3b_conv, 512, pad=1, kernel_size=3, stride=1, phase=phase)
	############ d4 ############
	net.d4a_pool = max_pool(net.d3c_conv, pad=0, kernel_size=2, stride=2)
	net.d4b_conv, net.d4b_bn, net.d4b_scale, net.d4b_relu = conv_bn_scale_relu(net.d4a_pool, 1024, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d4c_conv, net.d4c_bn, net.d4c_scale, net.d4c_relu = conv_bn_scale_relu(net.d4b_conv, 1024, pad=1, kernel_size=3, stride=1, phase=phase)

	############ u3 ############
	### a ### First Deconvolution
	net.u3a_dconv, net.u3a_bn, net.u3a_scale, net.u3a_relu = deconv_bn_scale_relu(net.d4c_conv, 512, pad=0, kernel_size=2, stride=2, phase=phase)
	### b ### Crop and Concat
	#net.u3b_crop = L.Crop(net.u3a_dconv, net.d3c_conv, axis=2, offset=0)
	#net.u3b_crop = crop(net.u3a_relu, net.d3c_relu)
	net.u3b_concat = L.Concat(net.u3a_dconv, net.d3c_conv, axis=1)
	### c ###
	net.u3c_conv, net.u3c_bn, net.u3c_scale, net.u3c_relu = conv_bn_scale_relu(net.u3b_concat, 512, pad=1, kernel_size=3, stride=1, phase=phase)
	### d ###
	net.u3d_conv, net.u3d_bn, net.u3d_scale, net.u3d_relu = conv_bn_scale_relu(net.u3c_conv, 512, pad=1, kernel_size=3, stride=1, phase=phase)

	############ u2 ############
	### a ### Second Deconvolution
	net.u2a_dconv, net.u2a_bn, net.u2a_scale, net.u2a_relu = deconv_bn_scale_relu(net.u3d_conv, 256, pad=0, kernel_size=2, stride=2, phase=phase)
	### b ### Crop and Concat
	#net.u2b_crop = L.Crop(net.u2a_dconv, net.d2c_conv, axis=2, offset=0)
	#net.u2b_crop = crop(net.u2a_relu, net.d2c_relu)
	net.u2b_concat = L.Concat(net.u2a_dconv, net.d2c_conv, axis=1)
	### c ###
	net.u2c_conv, net.u2c_bn, net.u2c_scale, net.u2c_relu = conv_bn_scale_relu(net.u2b_concat, 256, pad=1, kernel_size=3, stride=1, phase=phase)
	### d ###
	net.u2d_conv, net.u2d_bn, net.u2d_scale, net.u2d_relu = conv_bn_scale_relu(net.u2c_conv, 256, pad=1, kernel_size=3, stride=1, phase=phase)

	############ u1 ############
	### a ### Third Deconvolution
	net.u1a_dconv, net.u1a_bn, net.u1a_scale, net.u1a_relu = deconv_bn_scale_relu(net.u2d_relu, 128, pad=0, kernel_size=2, stride=2, phase=phase)
	### b ### Crop and Concat
	#net.u1b_crop = L.Crop(net.u1a_dconv, net.d1c_conv, axis=2, offset=0)
	#net.u1b_crop = crop(net.u1a_relu, net.d1c_relu)
	net.u1b_concat = L.Concat(net.u1a_dconv, net.d1c_conv, axis=1)
	### c ###
	net.u1c_conv, net.u1c_bn, net.u1c_scale, net.u1c_relu = conv_bn_scale_relu(net.u1b_concat, 128, pad=1, kernel_size=3, stride=1, phase=phase)
	### d ###
	net.u1d_conv, net.u1d_bn, net.u1d_scale, net.u1d_relu = conv_bn_scale_relu(net.u1c_conv, 128, pad=1, kernel_size=3, stride=1, phase=phase)

	############ u0 ############
	### a ### Fourth Deconvolution
	net.u0a_dconv, net.u0a_bn, net.u0a_scale, net.u0a_relu = deconv_bn_scale_relu(net.u1d_conv, 64, pad=0, kernel_size=2, stride=2, phase=phase)
	### b ### Crop and Concat
	#net.u0b_crop = L.Crop(net.u0a_dconv, net.d0c_conv, axis=2, offset=0)
	#net.u0b_crop = crop(net.u0a_relu, net.d0c_relu)
	net.u0b_concat = L.Concat(net.u0a_dconv, net.d0c_conv, axis=1)
	### c ###
	net.u0c_conv, net.u0c_bn, net.u0c_scale, net.u0c_relu = conv_bn_scale_relu(net.u0b_concat, 64, pad=1, kernel_size=3, stride=1, phase=phase)
	### d ###
	net.u0d_conv, net.u0d_bn, net.u0d_scale, net.u0d_relu = conv_bn_scale_relu(net.u0c_conv, 64, pad=1, kernel_size=3, stride=1, phase=phase)

	# ############ Score ############
	# net.score = L.Convolution(net.u0d_conv,
	# 	param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
	# 	num_output=num_class, pad=0, kernel_size=1, stride=1, 
	# 	# weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
	# 	weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
	# 	engine=engine)

	# ############ Loss ############
	# if phase == "train":
	# 	net.loss = L.SoftmaxWithLoss(net.score, net.label,
	# 		phase=0,
	# 		loss_weight=1,
	# 		loss_param=dict(ignore_label=ignore_label))
	# 	# net.loss = L.WeightedSoftmaxWithLoss(net.score, net.label, net.label_weight,
	# 	# 	phase=0,
	# 	# 	loss_weight=1,
	# 	# 	loss_param=dict(ignore_label=ignore_label))

	# ### loss 4 
	# net.score4 = L.Convolution(net.d4c_conv,
	# 	param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
	# 	num_output=num_class, pad=0, kernel_size=1, stride=1, 
	# 	weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	# net.upscore4 = L.Deconvolution(net.score4,
	# 	param=[dict(lr_mult=1, decay_mult=1)],
	# 	convolution_param=dict(num_output=num_class,  pad=8, kernel_size=32, stride=16,
	# 		weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	# if phase == "train":
	# 	net.loss4 = L.SoftmaxWithLoss(net.upscore4, net.label,
	# 		phase=0,
	# 		loss_weight=0.0625,
	# 		loss_param=dict(ignore_label=ignore_label))

	### loss 3
	net.score3 = L.Convolution(net.u3d_conv,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore3 = L.Deconvolution(net.score3,
		param=[dict(lr_mult=10, decay_mult=1)],
		convolution_param=dict(num_output=num_class,  pad=4, kernel_size=16, stride=8,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss3 = L.SoftmaxWithLoss(net.upscore3, net.label,
			phase=0,
			loss_weight=0.0625,
			loss_param=dict(ignore_label=ignore_label))

	### loss 2
	net.score2 = L.Convolution(net.u2d_conv,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore2 = L.Deconvolution(net.score2,
		param=[dict(lr_mult=10, decay_mult=1)],
		convolution_param=dict(num_output=num_class,  pad=2, kernel_size=8, stride=4,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss2 = L.SoftmaxWithLoss(net.upscore2, net.label,
			phase=0,
			loss_weight=0.125,
			loss_param=dict(ignore_label=ignore_label))

	### loss 1
	net.score1 = L.Convolution(net.u1d_conv,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore1 = L.Deconvolution(net.score1,
		param=[dict(lr_mult=10, decay_mult=1)],
		convolution_param=dict(num_output=num_class,  pad=1, kernel_size=4, stride=2,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss1 = L.SoftmaxWithLoss(net.upscore1, net.label,
			phase=0,
			loss_weight=0.25,
			loss_param=dict(ignore_label=ignore_label))

	### loss 0
	net.score = L.Convolution(net.u0d_conv,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	if phase == "train":
		net.loss = L.SoftmaxWithLoss(net.score, net.label,
			phase=0,
			loss_weight=1,
			loss_param=dict(ignore_label=ignore_label))

	# net.prob = L.Softmax(net.score)
	# net.slice10, net.slice11, net.slice12 = L.Slice(net.prob, slice_param=dict(axis=1, slice_point=[1,2]), ntop=3)
	# net.mask1 = L.Eltwise(net.slice11, net.slice12, eltwise_param=dict(operation=1))


	return net.to_proto()

def unet_2d_bn_weigted(dim_data, dim_label, num_class, ignore_label=255, phase="train"):
	net = caffe.NetSpec()
	############ d0 ############
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if phase == "train":
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		net.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))	

	net.d0b_conv, net.d0b_bn, net.d0b_scale, net.d0b_relu = conv_bn_scale_relu(net.data, 64, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d0c_conv, net.d0c_bn, net.d0c_scale, net.d0c_relu = conv_bn_scale_relu(net.d0b_conv, 64, pad=1, kernel_size=3, stride=1, phase=phase)
	############ d1 ############
	net.d1a_pool = max_pool(net.d0c_conv, pad=0, kernel_size=2, stride=2)
	net.d1b_conv, net.d1b_bn, net.d1b_scale, net.d1b_relu = conv_bn_scale_relu(net.d1a_pool, 128, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d1c_conv, net.d1c_bn, net.d1c_scale, net.d1c_relu = conv_bn_scale_relu(net.d1b_conv, 128, pad=1, kernel_size=3, stride=1, phase=phase)
	############ d2 ############
	net.d2a_pool = max_pool(net.d1c_conv, pad=0, kernel_size=2, stride=2)
	net.d2b_conv, net.d2b_bn, net.d2b_scale, net.d2b_relu = conv_bn_scale_relu(net.d2a_pool, 256, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d2c_conv, net.d2c_bn, net.d2c_scale, net.d2c_relu = conv_bn_scale_relu(net.d2b_conv, 256, pad=1, kernel_size=3, stride=1, phase=phase)
	############ d3 ############
	net.d3a_pool = max_pool(net.d2c_conv, pad=0, kernel_size=2, stride=2)
	net.d3b_conv, net.d3b_bn, net.d3b_scale, net.d3b_relu = conv_bn_scale_relu(net.d3a_pool, 512, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d3c_conv, net.d3c_bn, net.d3c_scale, net.d3c_relu = conv_bn_scale_relu(net.d3b_conv, 512, pad=1, kernel_size=3, stride=1, phase=phase)
	############ d4 ############
	net.d4a_pool = max_pool(net.d3c_conv, pad=0, kernel_size=2, stride=2)
	net.d4b_conv, net.d4b_bn, net.d4b_scale, net.d4b_relu = conv_bn_scale_relu(net.d4a_pool, 1024, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d4c_conv, net.d4c_bn, net.d4c_scale, net.d4c_relu = conv_bn_scale_relu(net.d4b_conv, 1024, pad=1, kernel_size=3, stride=1, phase=phase)

	############ u3 ############
	### a ### First Deconvolution
	net.u3a_dconv, net.u3a_bn, net.u3a_scale, net.u3a_relu = deconv_bn_scale_relu(net.d4c_conv, 512, pad=0, kernel_size=2, stride=2, phase=phase)
	### b ### Crop and Concat
	#net.u3b_crop = L.Crop(net.u3a_dconv, net.d3c_conv, axis=2, offset=0)
	#net.u3b_crop = crop(net.u3a_relu, net.d3c_relu)
	net.u3b_concat = L.Concat(net.u3a_dconv, net.d3c_conv, axis=1)
	### c ###
	net.u3c_conv, net.u3c_bn, net.u3c_scale, net.u3c_relu = conv_bn_scale_relu(net.u3b_concat, 512, pad=1, kernel_size=3, stride=1, phase=phase)
	### d ###
	net.u3d_conv, net.u3d_bn, net.u3d_scale, net.u3d_relu = conv_bn_scale_relu(net.u3c_conv, 512, pad=1, kernel_size=3, stride=1, phase=phase)

	############ u2 ############
	### a ### Second Deconvolution
	net.u2a_dconv, net.u2a_bn, net.u2a_scale, net.u2a_relu = deconv_bn_scale_relu(net.u3d_conv, 256, pad=0, kernel_size=2, stride=2, phase=phase)
	### b ### Crop and Concat
	#net.u2b_crop = L.Crop(net.u2a_dconv, net.d2c_conv, axis=2, offset=0)
	#net.u2b_crop = crop(net.u2a_relu, net.d2c_relu)
	net.u2b_concat = L.Concat(net.u2a_dconv, net.d2c_conv, axis=1)
	### c ###
	net.u2c_conv, net.u2c_bn, net.u2c_scale, net.u2c_relu = conv_bn_scale_relu(net.u2b_concat, 256, pad=1, kernel_size=3, stride=1, phase=phase)
	### d ###
	net.u2d_conv, net.u2d_bn, net.u2d_scale, net.u2d_relu = conv_bn_scale_relu(net.u2c_conv, 256, pad=1, kernel_size=3, stride=1, phase=phase)

	############ u1 ############
	### a ### Third Deconvolution
	net.u1a_dconv, net.u1a_bn, net.u1a_scale, net.u1a_relu = deconv_bn_scale_relu(net.u2d_relu, 128, pad=0, kernel_size=2, stride=2, phase=phase)
	### b ### Crop and Concat
	#net.u1b_crop = L.Crop(net.u1a_dconv, net.d1c_conv, axis=2, offset=0)
	#net.u1b_crop = crop(net.u1a_relu, net.d1c_relu)
	net.u1b_concat = L.Concat(net.u1a_dconv, net.d1c_conv, axis=1)
	### c ###
	net.u1c_conv, net.u1c_bn, net.u1c_scale, net.u1c_relu = conv_bn_scale_relu(net.u1b_concat, 128, pad=1, kernel_size=3, stride=1, phase=phase)
	### d ###
	net.u1d_conv, net.u1d_bn, net.u1d_scale, net.u1d_relu = conv_bn_scale_relu(net.u1c_conv, 128, pad=1, kernel_size=3, stride=1, phase=phase)

	############ u0 ############
	### a ### Fourth Deconvolution
	net.u0a_dconv, net.u0a_bn, net.u0a_scale, net.u0a_relu = deconv_bn_scale_relu(net.u1d_conv, 64, pad=0, kernel_size=2, stride=2, phase=phase)
	### b ### Crop and Concat
	#net.u0b_crop = L.Crop(net.u0a_dconv, net.d0c_conv, axis=2, offset=0)
	#net.u0b_crop = crop(net.u0a_relu, net.d0c_relu)
	net.u0b_concat = L.Concat(net.u0a_dconv, net.d0c_conv, axis=1)
	### c ###
	net.u0c_conv, net.u0c_bn, net.u0c_scale, net.u0c_relu = conv_bn_scale_relu(net.u0b_concat, 64, pad=1, kernel_size=3, stride=1, phase=phase)
	### d ###
	net.u0d_conv, net.u0d_bn, net.u0d_scale, net.u0d_relu = conv_bn_scale_relu(net.u0c_conv, 64, pad=1, kernel_size=3, stride=1, phase=phase)

	# ############ Score ############
	# net.score = L.Convolution(net.u0d_conv,
	# 	param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
	# 	num_output=num_class, pad=0, kernel_size=1, stride=1, 
	# 	# weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
	# 	weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
	# 	engine=engine)

	# ############ Loss ############
	# if phase == "train":
	# 	net.loss = L.SoftmaxWithLoss(net.score, net.label,
	# 		phase=0,
	# 		loss_weight=1,
	# 		loss_param=dict(ignore_label=ignore_label))
	# 	# net.loss = L.WeightedSoftmaxWithLoss(net.score, net.label, net.label_weight,
	# 	# 	phase=0,
	# 	# 	loss_weight=1,
	# 	# 	loss_param=dict(ignore_label=ignore_label))

	# ### loss 4 
	# net.score4 = L.Convolution(net.d4c_conv,
	# 	param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
	# 	num_output=num_class, pad=0, kernel_size=1, stride=1, 
	# 	weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	# net.upscore4 = L.Deconvolution(net.score4,
	# 	param=[dict(lr_mult=1, decay_mult=1)],
	# 	convolution_param=dict(num_output=num_class,  pad=8, kernel_size=32, stride=16,
	# 		weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	# if phase == "train":
	# 	net.loss4 = L.SoftmaxWithLoss(net.upscore4, net.label,
	# 		phase=0,
	# 		loss_weight=0.0625,
	# 		loss_param=dict(ignore_label=ignore_label))

	### loss 3
	net.score3 = L.Convolution(net.u3d_conv,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore3 = L.Deconvolution(net.score3,
		param=[dict(lr_mult=10, decay_mult=1)],
		convolution_param=dict(num_output=num_class,  pad=4, kernel_size=16, stride=8,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss3 = L.WeightedSoftmaxWithLoss(net.upscore3, net.label, net.label_weight,
			phase=0,
			loss_weight=0.0625,
			loss_param=dict(ignore_label=ignore_label))

	### loss 2
	net.score2 = L.Convolution(net.u2d_conv,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore2 = L.Deconvolution(net.score2,
		param=[dict(lr_mult=10, decay_mult=1)],
		convolution_param=dict(num_output=num_class,  pad=2, kernel_size=8, stride=4,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss2 = L.WeightedSoftmaxWithLoss(net.upscore2, net.label, net.label_weight,
			phase=0,
			loss_weight=0.125,
			loss_param=dict(ignore_label=ignore_label))

	### loss 1
	net.score1 = L.Convolution(net.u1d_conv,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore1 = L.Deconvolution(net.score1,
		param=[dict(lr_mult=10, decay_mult=1)],
		convolution_param=dict(num_output=num_class,  pad=1, kernel_size=4, stride=2,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss1 = L.WeightedSoftmaxWithLoss(net.upscore1, net.label, net.label_weight,
			phase=0,
			loss_weight=0.25,
			loss_param=dict(ignore_label=ignore_label))

	### loss 0
	net.score = L.Convolution(net.u0d_conv,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	if phase == "train":
		net.loss = L.WeightedSoftmaxWithLoss(net.score, net.label, net.label_weight,
			phase=0,
			loss_weight=1,
			loss_param=dict(ignore_label=ignore_label))
	else:
		net.prob = L.Softmax(net.score, axis=1)

	# net.prob = L.Softmax(net.score)
	# net.slice10, net.slice11, net.slice12 = L.Slice(net.prob, slice_param=dict(axis=1, slice_point=[1,2]), ntop=3)
	# net.mask1 = L.Eltwise(net.slice11, net.slice12, eltwise_param=dict(operation=1))
	# net.loss = L.WeightedSoftmaxWithLoss(net.score, net.label, net.label_weight,
		# 	phase=0,
		# 	loss_weight=1,
		# 	loss_param=dict(ignore_label=ignore_label))


	return net.to_proto()

def unet_3d(dim_data, dim_label, num_class, ignore_label=255, phase="train"):
	net = caffe.NetSpec()
	############ d0 ############
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if phase == "train":
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		# net.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))
	net.d0b_conv, net.d0b_relu = conv_relu(net.data, 32, pad=1, kernel_size=3, stride=1)
	net.d0c_conv, net.d0c_relu = conv_relu(net.d0b_conv, 32, pad=1, kernel_size=3, stride=1)
	############ d1 ############
	# net.d1a_pool = max_pool_nd(net.d0c_conv, pad=0, kernel_size=[2,2,1], stride=[2,2,1])
	net.d1a_pool = max_pool_nd(net.d0c_conv, pad=0, kernel_size=2, stride=2)
	net.d1b_conv, net.d1b_relu = conv_relu(net.d1a_pool, 64, pad=1, kernel_size=3, stride=1)
	net.d1c_conv, net.d1c_relu = conv_relu(net.d1b_conv, 64, pad=1, kernel_size=3, stride=1)
	############ d2 ############
	net.d2a_pool = max_pool_nd(net.d1c_conv, pad=0, kernel_size=2, stride=2)
	net.d2b_conv, net.d2b_relu = conv_relu(net.d2a_pool, 128, pad=1, kernel_size=3, stride=1)
	net.d2c_conv, net.d2c_relu = conv_relu(net.d2b_conv, 128, pad=1, kernel_size=3, stride=1)
	############ d3 ############
	net.d3a_pool = max_pool_nd(net.d2c_conv, pad=0, kernel_size=2, stride=2)
	net.d3b_conv, net.d3b_relu = conv_relu(net.d3a_pool, 256, pad=1, kernel_size=3, stride=1)
	net.d3c_conv, net.d3c_relu = conv_relu(net.d3b_conv, 256, pad=1, kernel_size=3, stride=1)

	############ u2 ############
	### a ### First Deconvolution
	net.u2a_dconv, net.u2a_relu = deconv_relu(net.d3c_conv, 128, pad=0, kernel_size=2, stride=2)
	### b ### Crop and Concat
	net.u2b_crop = L.Crop(net.u2a_dconv, net.d2c_conv, axis=2, offset=0)
	net.u2b_concat = L.Concat(net.u2b_crop, net.d2c_conv, axis=1)
	### c ###
	net.u2c_conv, net.u2c_relu = conv_relu(net.u2b_concat, 128, pad=1, kernel_size=3, stride=1)
	### d ###
	net.u2d_conv, net.u2d_relu = conv_relu(net.u2c_conv, 128, pad=1, kernel_size=3, stride=1)

	############ u1 ############
	### a ### Second Deconvolution
	net.u1a_dconv, net.u1a_relu = deconv_relu(net.u2d_conv, 64, pad=0, kernel_size=2, stride=2)
	### b ### Crop and Concat
	net.u1b_crop = L.Crop(net.u1a_dconv, net.d1c_conv, axis=2, offset=0)
	net.u1b_concat = L.Concat(net.u1b_crop, net.d1c_conv, axis=1)
	### c ###
	net.u1c_conv, net.u1c_relu = conv_relu(net.u1b_concat, 64, pad=1, kernel_size=3, stride=1)
	### d ###
	net.u1d_conv, net.u1d_relu = conv_relu(net.u1c_conv, 64, pad=1, kernel_size=3, stride=1)

	############ u0 ############
	### a ### Third Deconvolution
	#net.u0a_dconv, net.u0a_relu = deconv_relu(net.u1d_conv, 32, pad=0, kernel_size=[2,2,1], stride=[2,2,1])
	net.u0a_dconv, net.u0a_relu = deconv_relu(net.u1d_conv, 32, pad=0, kernel_size=2, stride=2)
	### b ### Crop and Concat
	net.u0b_crop = L.Crop(net.u0a_dconv, net.d0c_conv, axis=2, offset=0)
	net.u0b_concat = L.Concat(net.u0b_crop, net.d0c_conv, axis=1)
	### c ###
	net.u0c_conv, net.u0c_relu = conv_relu(net.u0b_concat, 32, pad=1, kernel_size=3, stride=1)
	### d ###
	net.u0d_conv, net.u0d_relu = conv_relu(net.u0c_conv, 32, pad=1, kernel_size=3, stride=1)

	############ Score ############
	net.score = L.Convolution(net.u0d_conv,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=engine)

	############ Loss ############
	if phase == "train":
		net.loss = L.SoftmaxWithLoss(net.score, net.label,
			phase=0,
			loss_weight=1,
			loss_param=dict(ignore_label=ignore_label))
		# net.loss = L.WeightedSoftmaxWithLoss(net.score, net.label, net.label_weight,
		# 	phase=0,
		# 	loss_weight=1,
		# 	loss_param=dict(ignore_label=ignore_label))

	return net.to_proto()

def unet_3d_bn(dim_data, dim_label, num_class, ignore_label=255, phase="train"):
	net = caffe.NetSpec()
	############ d0 ############
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if phase == "train":
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		# net.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))	
	net.d0b_conv, net.d0b_bn, net.d0b_scale, net.d0b_relu = conv_bn_scale_relu(net.data, 32, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d0c_conv, net.d0c_bn, net.d0c_scale, net.d0c_relu = conv_bn_scale_relu(net.d0b_conv, 32, pad=1, kernel_size=3, stride=1, phase=phase)
	############ d1 ############
	# net.d1a_pool = max_pool_nd(net.d0c_relu, pad=0, kernel_size=[2,2,1], stride=[2,2,1])
	net.d1a_pool = max_pool_nd(net.d0c_conv, pad=0, kernel_size=2, stride=2)
	net.d1b_conv, net.d1b_bn, net.d1b_scale, net.d1b_relu = conv_bn_scale_relu(net.d1a_pool, 64, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d1c_conv, net.d1c_bn, net.d1c_scale, net.d1c_relu = conv_bn_scale_relu(net.d1b_conv, 64, pad=1, kernel_size=3, stride=1, phase=phase)
	############ d2 ############
	net.d2a_pool = max_pool_nd(net.d1c_conv, pad=0, kernel_size=2, stride=2)
	net.d2b_conv, net.d2b_bn, net.d2b_scale, net.d2b_relu = conv_bn_scale_relu(net.d2a_pool, 128, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d2c_conv, net.d2c_bn, net.d2c_scale, net.d2c_relu = conv_bn_scale_relu(net.d2b_conv, 128, pad=1, kernel_size=3, stride=1, phase=phase)
	############ d3 ############
	net.d3a_pool = max_pool_nd(net.d2c_relu, pad=0, kernel_size=2, stride=2)
	net.d3b_conv, net.d3b_bn, net.d3b_scale, net.d3b_relu = conv_bn_scale_relu(net.d3a_pool, 256, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d3c_conv, net.d3c_bn, net.d3c_scale, net.d3c_relu = conv_bn_scale_relu(net.d3b_conv, 256, pad=1, kernel_size=3, stride=1, phase=phase)

	############ u2 ############
	### a ### First Deconvolution
	net.u2a_dconv, net.u2a_bn, net.u2a_scale, net.u2a_relu = deconv_bn_scale_relu(net.d3c_conv, 128, pad=0, kernel_size=2, stride=2, phase=phase)
	### b ### Crop and Concat
	net.u2b_crop = L.Crop(net.u2a_dconv, net.d2c_conv, axis=2, offset=0)
	net.u2b_concat = L.Concat(net.u2b_crop, net.d2c_conv, axis=1)
	### c ###
	net.u2c_conv, net.u2c_bn, net.u2c_scale, net.u2c_relu = conv_bn_scale_relu(net.u2b_concat, 128, pad=1, kernel_size=3, stride=1, phase=phase)
	### d ###
	net.u2d_conv, net.u2d_bn, net.u2d_scale, net.u2d_relu = conv_bn_scale_relu(net.u2c_conv, 128, pad=1, kernel_size=3, stride=1, phase=phase)

	############ u1 ############
	### a ### Second Deconvolution
	net.u1a_dconv, net.u1a_bn, net.u1a_scale, net.u1a_relu = deconv_bn_scale_relu(net.u2d_conv, 64, pad=0, kernel_size=2, stride=2, phase=phase)
	### b ### Crop and Concat
	net.u1b_crop = L.Crop(net.u1a_dconv, net.d1c_conv, axis=2, offset=0)
	net.u1b_concat = L.Concat(net.u1b_crop, net.d1c_conv, axis=1)
	### c ###
	net.u1c_conv, net.u1c_bn, net.u1c_scale, net.u1c_relu = conv_bn_scale_relu(net.u1b_concat, 64, pad=1, kernel_size=3, stride=1, phase=phase)
	### d ###
	net.u1d_conv, net.u1d_bn, net.u1d_scale, net.u1d_relu = conv_bn_scale_relu(net.u1c_conv, 64, pad=1, kernel_size=3, stride=1, phase=phase)

	############ u0 ############
	### a ### Third Deconvolution
	# net.u0a_dconv, net.u0a_bn, net.u0a_scale, net.u0a_relu = deconv_bn_scale_relu(net.u1d_relu, 32, pad=0, kernel_size=[2,2,1], stride=[2,2,1], phase=phase)
	net.u0a_dconv, net.u0a_bn, net.u0a_scale, net.u0a_relu = deconv_bn_scale_relu(net.u1d_conv, 32, pad=0, kernel_size=2, stride=2, phase=phase)
	### b ### Crop and Concat
	net.u0b_crop = L.Crop(net.u0a_dconv, net.d0c_conv, axis=2, offset=0)
	net.u0b_concat = L.Concat(net.u0b_crop, net.d0c_conv, axis=1)
	### c ###
	net.u0c_conv, net.u0c_bn, net.u0c_scale, net.u0c_relu = conv_bn_scale_relu(net.u0b_concat, 32, pad=1, kernel_size=3, stride=1, phase=phase)
	### d ###
	net.u0d_conv, net.u0d_bn, net.u0d_scale, net.u0d_relu = conv_bn_scale_relu(net.u0c_conv, 32, pad=1, kernel_size=3, stride=1, phase=phase)

	# ############ Score ############
	# net.score = L.Convolution(net.u0d_conv,
	# 	param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
	# 	num_output=num_class, pad=0, kernel_size=1, stride=1, 
	# 	weight_filler=dict(type='msra', std=0.001), bias_filler=dict(type='constant', value=0),
	# 	engine=engine)

	# ############ Loss ############
	# # if phase == "train":
	# # 	net.loss = L.WeightedSoftmaxWithLoss(net.score, net.label, net.label_weight,
	# # 		phase=0,
	# # 		loss_weight=1,
	# # 		loss_param=dict(ignore_label=ignore_label))
	# if phase == "train":
	# 	net.loss = L.SoftmaxWithLoss(net.score, net.label,
	# 		phase=0,
	# 		loss_weight=1,
	# 		loss_param=dict(ignore_label=ignore_label))
	# ### loss 3
	# net.score3 = L.Convolution(net.u3d_conv,
	# 	param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
	# 	num_output=num_class, pad=0, kernel_size=1, stride=1, 
	# 	weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	# net.upscore3 = L.Deconvolution(net.score3,
	# 	param=[dict(lr_mult=1, decay_mult=1)],
	# 	convolution_param=dict(num_output=num_class,  pad=4, kernel_size=16, stride=8,
	# 		weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	# if phase == "train":
	# 	net.loss3 = L.SoftmaxWithLoss(net.upscore3, net.label,
	# 		phase=0,
	# 		loss_weight=0.0625,
	# 		loss_param=dict(ignore_label=ignore_label))

	### loss 2
	#net.score2 = L.Convolution(net.u2d_conv,
	#	param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
	#	num_output=num_class, pad=0, kernel_size=1, stride=1, 
	#	weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	#net.upscore2 = L.Deconvolution(net.score2,
	#	param=[dict(lr_mult=1, decay_mult=1)],
	#	convolution_param=dict(num_output=num_class,  pad=2, kernel_size=8, stride=4,
	#		weight_filler=dict(type='msra'), bias_term=0, engine=engine))
	#if phase == "train":
	#	net.loss2 = L.SoftmaxWithLoss(net.upscore2, net.label,
	#		phase=0,
	#		loss_weight=0.25,
	#		loss_param=dict(ignore_label=ignore_label))

	### loss 1
	#net.score1 = L.Convolution(net.u1d_conv,
	#	param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
	#	num_output=num_class, pad=0, kernel_size=1, stride=1, 
	#	weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	#net.upscore1 = L.Deconvolution(net.score1,
	#	param=[dict(lr_mult=1, decay_mult=1)],
	#	convolution_param=dict(num_output=num_class,  pad=1, kernel_size=4, stride=2,
	#		weight_filler=dict(type='msra'), bias_term=0, engine=engine))
	#if phase == "train":
	#	net.loss1 = L.SoftmaxWithLoss(net.upscore1, net.label,
	#		phase=0,
	#		loss_weight=0.25,
	#		loss_param=dict(ignore_label=ignore_label))

	### loss 0
	net.score = L.Convolution(net.u0d_conv,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	if phase == "train":
		net.loss = L.SoftmaxWithLoss(net.score, net.label,
			phase=0,
			loss_weight=1,
			loss_param=dict(ignore_label=ignore_label))

	return net.to_proto()


def make_unet(net, dim_data, dim_label, num_class, prototxt_train, prototxt_test):
	# register net
	__nets = ['unet_2d', 'unet_2d_bn', 'unet_3d', 'unet_3d_bn', 'unet_rcu_2d', 'unet_2rcu_2d', 'unet_2d_bn_8s', 'unet_rcu_2d_bn',
	'unet_2d_bn_mask', 'unet_2d_bn_weigted']
	assert net in __nets, 'Unknown net: {}'.format(net)

	global engine, ignore_label
	engine = 2
	ignore_label = 255

	if net == 'unet_2d':
		train_net = unet_2d(dim_data, dim_label, num_class, ignore_label, phase="train")
		test_net = unet_2d(dim_data, dim_label, num_class, ignore_label, phase="test")
	if net == 'unet_2d_bn':
		train_net = unet_2d_bn(dim_data, dim_label, num_class, ignore_label, phase="train")
		test_net = unet_2d_bn(dim_data, dim_label, num_class, ignore_label, phase="test")
	if net == 'unet_3d':
		train_net = unet_3d(dim_data, dim_label, num_class, ignore_label, phase="train")
		test_net = unet_3d(dim_data, dim_label, num_class, ignore_label, phase="test")
	if net == 'unet_3d_bn':
		train_net = unet_3d_bn(dim_data, dim_label, num_class, ignore_label, phase="train")
		test_net = unet_3d_bn(dim_data, dim_label, num_class, ignore_label, phase="test")
	if net == 'unet_rcu_2d':
		train_net = unet_rcu_2d(dim_data, dim_label, num_class, ignore_label, phase="train")
		test_net = unet_rcu_2d(dim_data, dim_label, num_class, ignore_label, phase="test")

	if net == 'unet_2rcu_2d':
		train_net = unet_2rcu_2d(dim_data, dim_label, num_class, ignore_label, phase="train")
		test_net = unet_2rcu_2d(dim_data, dim_label, num_class, ignore_label, phase="test")

	if net == 'unet_2d_bn_8s':
		train_net = unet_2d_bn_8s(dim_data, dim_label, num_class, ignore_label, phase="train")
		test_net = unet_2d_bn_8s(dim_data, dim_label, num_class, ignore_label, phase="test")

	if net == 'unet_rcu_2d_bn':
		train_net = unet_rcu_2d_bn(dim_data, dim_label, num_class, ignore_label, phase="train")
		test_net = unet_rcu_2d_bn(dim_data, dim_label, num_class, ignore_label, phase="test")

	if net == 'unet_2d_bn_mask':
		train_net = unet_2d_bn_mask(dim_data, dim_label, num_class, ignore_label, phase="train")
		test_net = unet_2d_bn_mask(dim_data, dim_label, num_class, ignore_label, phase="test")

	if net == 'unet_2d_bn_weigted':
		train_net = unet_2d_bn_weigted(dim_data, dim_label, num_class, ignore_label, phase="train")
		test_net = unet_2d_bn_weigted(dim_data, dim_label, num_class, ignore_label, phase="test")




	with open(prototxt_train, 'w') as f:
		f.write(str(train_net))

	with open(prototxt_test, 'w') as f:
		f.write(str(test_net))