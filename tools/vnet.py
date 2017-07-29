import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop # crop(net.upscore, net.data) automatically calculate the axis and offset

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
def conv_bn_scale_relu(bottom, num_output, pad=0, kernel_size=3, stride=1):
	conv = L.Convolution(bottom,
		num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
		# weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], engine=engine)
	bn = L.BatchNorm(conv, in_place=True, use_global_stats=use_global_stats, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	scale = L.Scale(bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	relu = L.ReLU(scale, in_place=True, engine=engine)
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
def deconv_bn_scale_relu(bottom, num_output, pad=0, kernel_size=3, stride=1, phase='train', engine=2):
	deconv = L.Deconvolution(bottom,
		param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=num_output,  pad=pad, kernel_size=kernel_size, stride=stride,
			# weight_filler=dict(type='gaussian', std=0.001), bias_term=0,
			weight_filler=dict(type='msra'), bias_term=0,
			engine=engine))
	bn = L.BatchNorm(deconv, in_place=True, use_global_stats=use_global_stats, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	scale = L.Scale(bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	relu = L.ReLU(scale, in_place=True, engine=engine)
	# relu = L.PReLU(scale, in_place=True)
	return deconv, bn, scale, relu

def residual_block(bottom, num_output, pad=0, kernel_size=3, stride=1):
	# path 1
	conv1a = L.Convolution(bottom,
		num_output=num_output, pad=0, kernel_size=1, stride=1,
		# weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], engine=engine)

	# path 2
	conv2a = L.Convolution(bottom,
		num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
		# weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], engine=engine)
	relu2a = L.ReLU(conv2a, in_place=True, engine=engine)
	conv2b = L.Convolution(conv2a,
		num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
		# weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], engine=engine)

	# merge
	eltw = L.Eltwise(conv1a, conv2b, eltwise_param=dict(operation=1))
	eltw_relu = L.ReLU(eltw, in_place=True, engine=engine)

	return conv1a, conv2a, relu2a, conv2b, eltw, eltw_relu
	


############ ############ ############ ############ ############ ###########
def vnet_2d(dim_data, dim_label, num_class, ignore_label=255, phase="train"):
	net = caffe.NetSpec()
	############ d0 ############
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if phase == "train":
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		# net.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))
	net.d0b_conv, net.d0b_relu = conv_relu(net.data, 64, pad=1, kernel_size=3, stride=1)
	net.d0c_conv, net.d0c_relu = conv_relu(net.d0b_conv, 64, pad=1, kernel_size=3, stride=1)
	############ d1 ############
	net.d1a_pool = L.Pooling(net.d0c_conv, pool=P.Pooling.MAX, pad=0, kernel_size=2, stride=2)
	net.d1b_conv1a, net.d1b_conv2a, net.d1b_relu2a, net.d1b_conv2b, net.d1b_eltw, net.d1b_eltw_relu = residual_block(net.d1a_pool, 128, pad=1, kernel_size=3, stride=1)
	############ d2 ############
	net.d2a_pool = L.Pooling(net.d1b_eltw, pool=P.Pooling.MAX, pad=0, kernel_size=2, stride=2)
	net.d2b_conv1a, net.d2b_conv2a, net.d2b_relu2a, net.d2b_conv2b, net.d2b_eltw, net.d2b_eltw_relu = residual_block(net.d2a_pool, 256, pad=1, kernel_size=3, stride=1)
	############ d3 ############
	net.d3a_pool = L.Pooling(net.d2b_eltw, pool=P.Pooling.MAX, pad=0, kernel_size=2, stride=2)
	net.d3b_conv1a, net.d3b_conv2a, net.d3b_relu2a, net.d3b_conv2b, net.d3b_eltw, net.d3b_eltw_relu = residual_block(net.d3a_pool, 512, pad=1, kernel_size=3, stride=1)
	############ d4 ############
	net.d4a_pool = L.Pooling(net.d3b_eltw, pool=P.Pooling.MAX, pad=0, kernel_size=2, stride=2)
	net.d4b_conv1a, net.d4b_conv2a, net.d4b_relu2a, net.d4b_conv2b, net.d4b_eltw, net.d4b_eltw_relu = residual_block(net.d4a_pool, 1024, pad=1, kernel_size=3, stride=1)
	

	############ u3 ############
	### a ### First Deconvolution
	net.u3a_dconv, net.u3a_relu = deconv_relu(net.d4b_eltw, 512, pad=1, kernel_size=4, stride=2)
	### b ### Crop and Concat
	#net.u3b_crop = L.Crop(net.u3a_relu, net.d3c_relu, axis=2, offset=0)
	net.u3b_concat = L.Concat(net.u3a_dconv, net.d3b_eltw, axis=1)
	### c ###
	net.u3c_conv1a, net.u3c_conv2a, net.u3c_relu2a, net.u3c_conv2b, net.u3c_eltw, net.u3c_eltw_relu = residual_block(net.u3b_concat, 512, pad=1, kernel_size=3, stride=1)


	############ u2 ############
	### a ### Second Deconvolution
	net.u2a_dconv, net.u2a_relu = deconv_relu(net.u3c_eltw, 256, pad=1, kernel_size=4, stride=2)
	### b ### Crop and Concat
	#net.u2b_crop = L.Crop(net.u2a_relu, net.d2c_relu, axis=2, offset=0)
	net.u2b_concat = L.Concat(net.u2a_dconv, net.d2b_eltw, axis=1)
	### c ###
	net.u2c_conv1a, net.u2c_conv2a, net.u2c_relu2a, net.u2c_conv2b, net.u2c_eltw, net.u2c_eltw_relu = residual_block(net.u2b_concat, 256, pad=1, kernel_size=3, stride=1)


	############ u1 ############
	### a ### Third Deconvolution
	net.u1a_dconv, net.u1a_relu = deconv_relu(net.u2c_eltw, 128, pad=1, kernel_size=4, stride=2)
	### b ### Crop and Concat
	#net.u1b_crop = L.Crop(net.u1a_relu, net.d1c_relu, axis=2, offset=0)
	net.u1b_concat = L.Concat(net.u1a_dconv, net.d1b_eltw, axis=1)
	### c ###
	net.u1c_conv1a, net.u1c_conv2a, net.u1c_relu2a, net.u1c_conv2b, net.u1c_eltw, net.u1c_eltw_relu = residual_block(net.u1b_concat, 128, pad=1, kernel_size=3, stride=1)


	############ u0 ############
	### a ### Fourth Deconvolution
	net.u0a_dconv, net.u0a_relu = deconv_relu(net.u1c_eltw, 64, pad=1, kernel_size=4, stride=2)
	### b ### Crop and Concat
	#net.u0b_crop = L.Crop(net.u0a_relu, net.d0c_relu, axis=2, offset=0)
	net.u0b_concat = L.Concat(net.u0a_dconv, net.d0c_conv, axis=1)
	### c ###
	# net.u0c_conv, net.u0c_relu = conv_relu(net.u0b_concat, 64, pad=1, kernel_size=3, stride=1)
	# ### d ###
	# net.u0d_conv, net.u0d_relu = conv_relu(net.u0c_conv, 64, pad=1, kernel_size=3, stride=1)
	net.u0c_conv1a, net.u0c_conv2a, net.u0c_relu2a, net.u0c_conv2b, net.u0c_eltw, net.u0c_eltw_relu = residual_block(net.u0b_concat, 64, pad=1, kernel_size=3, stride=1)


	############ Score ############
	# net.score = L.Convolution(net.u0c_eltw,
	# 	param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
	# 	num_output=num_class, pad=0, kernel_size=1, stride=1, 
	# 	#weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
	# 	weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
	# 	engine=2)

	############ Loss ############
	# if phase == "train":
	# 	net.loss = L.SoftmaxWithLoss(net.score, net.label,
	# 		phase=0,
	# 		loss_weight=1,
	# 		loss_param=dict(ignore_label=ignore_label))


	### loss 4 
	# net.score4 = L.Convolution(net.d4b_eltw,
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
	net.score3 = L.Convolution(net.u3c_eltw,
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
			loss_weight=0.125,
			loss_param=dict(ignore_label=ignore_label))

	### loss 2
	net.score2 = L.Convolution(net.u2c_eltw,
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
	net.score1 = L.Convolution(net.u1c_eltw,
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
	net.score = L.Convolution(net.u0c_eltw,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	if phase == "train":
		net.loss = L.SoftmaxWithLoss(net.score, net.label,
			phase=0,
			loss_weight=0.5,
			loss_param=dict(ignore_label=ignore_label))


	return net.to_proto()

def vnet_2d_bn(dim_data, dim_label, num_class, ignore_label=255, phase="train"):
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

	############ Score ############
	net.score = L.Convolution(net.u0d_conv,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		# weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=2)

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



engine = 2
ignore_label = -1

def make_vnet(net, dim_data, dim_label, num_class, prototxt_train, prototxt_test):
	# register net
	__nets = ['vnet_2d', 'vnet_2d_bn']
	assert net in __nets, 'Unknown net: {}'.format(net)

	global engine, use_global_stats, ignore_label
	engine = 2
	ignore_label = 255


	if net == 'vnet_2d':
		train_net = vnet_2d(dim_data, dim_label, num_class, ignore_label, phase="train")
		test_net = vnet_2d(dim_data, dim_label, num_class, ignore_label, phase="test")
	if net == 'vnet_2d_bn':
		train_net = vnet_2d_bn(dim_data, dim_label, num_class, ignore_label, phase="train")
		test_net = vnet_2d_bn(dim_data, dim_label, num_class, ignore_label, phase="test")


	with open(prototxt_train, 'w') as f:
		f.write(str(train_net))

	with open(prototxt_test, 'w') as f:
		f.write(str(test_net))