import caffe
from caffe import layers as L, params as P, to_proto
from caffe.coord_map import crop # crop(net.upscore, net.data) automatically calculate the axis and offset

def bn_relu_conv(bottom, num_output, pad=0, kernel_size=3, stride=1):
	bn = L.BatchNorm(bottom, in_place=False, use_global_stats=use_global_stats, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	scale = L.Scale(bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	relu = L.ReLU(scale, in_place=True, engine=engine)
	conv = L.Convolution(relu, num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],		
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=engine)
	return bn, scale, relu, conv

def conv_bn(bottom, num_output, pad=0, kernel_size=3, stride=1):
	conv = L.Convolution(bottom, num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],		
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=engine)
	bn = L.BatchNorm(conv, in_place=True, use_global_stats=use_global_stats, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	scale = L.Scale(bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	return conv, bn, scale

def conv_bn_relu(bottom, num_output, pad=0, kernel_size=3, stride=1):
	conv = L.Convolution(bottom, num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],		
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=engine)
	bn = L.BatchNorm(conv, in_place=True, use_global_stats=use_global_stats, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	scale = L.Scale(bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	relu = L.ReLU(scale, in_place=True, engine=engine)
	return conv, bn, scale, relu

def deconv_bn_relu(bottom, num_output, pad=0, kernel_size=3, stride=1):
	deconv = L.Deconvolution(bottom, param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=num_output,  pad=pad, kernel_size=kernel_size, stride=stride,
			weight_filler=dict(type='msra'), bias_term=0,
			engine=engine))
	bn = L.BatchNorm(deconv, in_place=True, use_global_stats=use_global_stats, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	scale = L.Scale(bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	relu = L.ReLU(scale, in_place=True, engine=engine)
	return deconv, bn, scale, relu

def add_layer(bottom1, bottom2, num_output):
	conv, bn, scale = conv_bn(bottom1, num_output=num_output, pad=0, kernel_size=1, stride=1)
	eltw = L.Eltwise(conv, bottom2, eltwise_param=dict(operation=1))
	rule = L.ReLU(eltw, in_place=True, engine=engine)
	return conv, bn, scale, eltw, rule

def conv1_conv2_add_bn_relu(bottom1, bottom2, num_output, pad=0, kernel_size=3, stride=1):
	conv1 = L.Convolution(bottom1, num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],		
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=engine)
	conv2 = L.Convolution(bottom2, num_output=num_output, pad=0, kernel_size=1, stride=1,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],		
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=engine)
	eltw = L.Eltwise(conv1, conv2, eltwise_param=dict(operation=1))
	bn = L.BatchNorm(eltw, in_place=True, use_global_stats=use_global_stats, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	scale = L.Scale(bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	relu = L.ReLU(scale, in_place=True, engine=engine)
	
	return conv1, conv2, eltw, bn, scale, relu


def max_pool(bottom, pad=0, kernel_size=2, stride=2):
	return L.Pooling(bottom, pool=P.Pooling.MAX, pad=pad, kernel_size=kernel_size, stride=stride, engine=engine)
############ ############
def uvnet_2d_bn(dim_data, dim_label, num_class, phase='train'):
	net = caffe.NetSpec()
	############ d0 ############
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if phase == "train":
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		# net.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))	
	net.d0b_conv, net.d0b_bn, net.d0b_scale, net.d0b_relu = conv_bn_relu(net.data, 64, pad=1, kernel_size=3, stride=1)
	net.d0c_conv, net.d0c_bn, net.d0c_scale, net.d0c_relu = conv_bn_relu(net.d0b_conv, 64, pad=1, kernel_size=3, stride=1)
	############ d1 ############
	net.d1a_pool = max_pool(net.d0c_conv, pad=0, kernel_size=2, stride=2)
	net.d1b_conv, net.d1b_bn, net.d1b_scale, net.d1b_relu = conv_bn_relu(net.d1a_pool, 128, pad=1, kernel_size=3, stride=1)
	net.d1c_conv, net.d1c_bn, net.d1c_scale, net.d1c_relu = conv_bn_relu(net.d1b_conv, 128, pad=1, kernel_size=3, stride=1)
	#net.d1d_conv, net.d1d_bn, net.d1d_scale = conv_bn(net.d1c_conv, 128, pad=1, kernel_size=3, stride=1)
	net.d1e_conv, net.d1e_bn, net.d1e_scale, net.d1e_eltw, net.d1e_relu = add_layer(net.d1a_pool, net.d1c_conv, 128)
	############ d2 ############
	net.d2a_pool = max_pool(net.d1e_relu, pad=0, kernel_size=2, stride=2)
	net.d2b_conv, net.d2b_bn, net.d2b_scale, net.d2b_relu = conv_bn_relu(net.d2a_pool, 256, pad=1, kernel_size=3, stride=1)
	net.d2c_conv, net.d2c_bn, net.d2c_scale, net.d2c_relu = conv_bn_relu(net.d2b_conv, 256, pad=1, kernel_size=3, stride=1)
	net.d2d_conv, net.d2d_bn, net.d2d_scale = conv_bn(net.d2c_conv, 256, pad=1, kernel_size=3, stride=1)
	net.d2e_conv, net.d2e_bn, net.d2e_scale, net.d2e_eltw, net.d2e_relu = add_layer(net.d2a_pool, net.d2d_conv, 256)
	############ d3 ############
	net.d3a_pool = max_pool(net.d2e_relu, pad=0, kernel_size=2, stride=2)
	net.d3b_conv, net.d3b_bn, net.d3b_scale, net.d3b_relu = conv_bn_relu(net.d3a_pool, 512, pad=1, kernel_size=3, stride=1)
	net.d3c_conv, net.d3c_bn, net.d3c_scale, net.d3c_relu = conv_bn_relu(net.d3b_conv, 512, pad=1, kernel_size=3, stride=1)
	net.d3d_conv, net.d3d_bn, net.d3d_scale = conv_bn(net.d3c_conv, 512, pad=1, kernel_size=3, stride=1)
	net.d3e_conv, net.d3e_bn, net.d3e_scale, net.d3e_eltw, net.d3e_relu = add_layer(net.d3a_pool, net.d3d_conv, 512)
	############ d4 ############
	net.d4a_pool = max_pool(net.d3e_relu, pad=0, kernel_size=2, stride=2)
	net.d4b_conv, net.d4b_bn, net.d4b_scale, net.d4b_relu = conv_bn_relu(net.d4a_pool, 1024, pad=1, kernel_size=3, stride=1)
	net.d4c_conv, net.d4c_bn, net.d4c_scale, net.d4c_relu = conv_bn_relu(net.d4b_conv, 1024, pad=1, kernel_size=3, stride=1)
	net.d4d_conv, net.d4d_bn, net.d4d_scale = conv_bn(net.d4c_conv, 1024, pad=1, kernel_size=3, stride=1)
	net.d4e_conv, net.d4e_bn, net.d4e_scale, net.d4e_eltw, net.d4e_relu = add_layer(net.d4a_pool, net.d4d_conv, 1024)
	############ u3 ############
	### a ### First Deconvolution
	net.u3a_dconv, net.u3a_bn, net.u3a_scale, net.u3a_relu = deconv_bn_relu(net.d4e_relu, 512, pad=0, kernel_size=2, stride=2)
	net.u3b_concat = L.Concat(net.u3a_dconv, net.d3e_relu, axis=1)
	net.u3c_conv, net.u3c_bn, net.u3c_scale, net.u3c_relu = conv_bn_relu(net.u3b_concat, 512, pad=1, kernel_size=3, stride=1)
	net.u3d_conv, net.u3d_bn, net.u3d_scale, net.u3d_relu = conv_bn_relu(net.u3c_conv, 512, pad=1, kernel_size=3, stride=1)
	net.u3e_conv, net.u3e_bn, net.u3e_scale = conv_bn(net.u3d_conv, 512, pad=1, kernel_size=3, stride=1)
	net.u3f_conv, net.u3f_bn, net.u3f_scale, net.u3f_eltw, net.u3f_relu = add_layer(net.u3b_concat, net.u3e_conv, 512)
	############ u2 ############
	### a ### Second Deconvolution
	net.u2a_dconv, net.u2a_bn, net.u2a_scale, net.u2a_relu = deconv_bn_relu(net.u3f_relu, 256, pad=0, kernel_size=2, stride=2)
	net.u2b_concat = L.Concat(net.u2a_dconv, net.d2e_relu, axis=1)
	net.u2c_conv, net.u2c_bn, net.u2c_scale, net.u2c_relu = conv_bn_relu(net.u2b_concat, 256, pad=1, kernel_size=3, stride=1)
	net.u2d_conv, net.u2d_bn, net.u2d_scale, net.u2d_relu = conv_bn_relu(net.u2c_conv, 256, pad=1, kernel_size=3, stride=1)
	net.u2e_conv, net.u2e_bn, net.u2e_scale = conv_bn(net.u2d_conv, 256, pad=1, kernel_size=3, stride=1)
	net.u2f_conv, net.u2f_bn, net.u2f_scale, net.u2f_eltw, net.u2f_relu = add_layer(net.u2b_concat, net.u2e_conv, 256)
	############ u1 ############
	### a ### Third Deconvolution
	net.u1a_dconv, net.u1a_bn, net.u1a_scale, net.u1a_relu = deconv_bn_relu(net.u2f_relu, 128, pad=0, kernel_size=2, stride=2)
	net.u1b_concat = L.Concat(net.u1a_dconv, net.d1e_relu, axis=1)
	net.u1c_conv, net.u1c_bn, net.u1c_scale, net.u1c_relu = conv_bn_relu(net.u1b_concat, 128, pad=1, kernel_size=3, stride=1)
	# net.u1d_conv, net.u1d_bn, net.u1d_scale, net.u1d_relu = conv_bn_relu(net.u1c_conv, 128, pad=1, kernel_size=3, stride=1)
	net.u1e_conv, net.u1e_bn, net.u1e_scale = conv_bn(net.u1c_conv, 128, pad=1, kernel_size=3, stride=1)
	net.u1f_conv, net.u1f_bn, net.u1f_scale, net.u1f_eltw, net.u1f_relu = add_layer(net.u1b_concat, net.u1e_conv, 128)
	############ u0 ############
	### a ### Fourth Deconvolution
	net.u0a_dconv, net.u0a_bn, net.u0a_scale, net.u0a_relu = deconv_bn_relu(net.u1f_relu, 64, pad=0, kernel_size=2, stride=2)
	net.u0b_concat = L.Concat(net.u0a_dconv, net.d0c_conv, axis=1)
	net.u0c_conv, net.u0c_bn, net.u0c_scale, net.u0c_relu = conv_bn_relu(net.u0b_concat, 64, pad=1, kernel_size=3, stride=1)
	net.u0d_conv, net.u0d_bn, net.u0d_scale = conv_bn(net.u0c_conv, 64, pad=1, kernel_size=3, stride=1)
	net.u0e_conv, net.u0e_bn, net.u0e_scale, net.u0e_eltw, net.u0e_relu = add_layer(net.u0b_concat, net.u0d_conv, 64)
	############ score ###########
	net.score3 = L.Convolution(net.u3f_relu,
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
	net.score2 = L.Convolution(net.u2f_relu,
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
	net.score1 = L.Convolution(net.u1f_relu,
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
	net.score = L.Convolution(net.u0e_relu,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],	
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

def uvnet_2d_bn_weigted(dim_data, dim_label, num_class, phase='train'):
	net = caffe.NetSpec()
	############ d0 ############
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if phase == "train":
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		net.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))	
	net.d0b_conv, net.d0b_bn, net.d0b_scale, net.d0b_relu = conv_bn_relu(net.data, 64, pad=1, kernel_size=3, stride=1)
	net.d0c_conv, net.d0c_bn, net.d0c_scale, net.d0c_relu = conv_bn_relu(net.d0b_conv, 64, pad=1, kernel_size=3, stride=1)
	############ d1 ############
	net.d1a_pool = max_pool(net.d0c_conv, pad=0, kernel_size=2, stride=2)
	net.d1b_conv, net.d1b_bn, net.d1b_scale, net.d1b_relu = conv_bn_relu(net.d1a_pool, 128, pad=1, kernel_size=3, stride=1)
	net.d1c_conv, net.d1c_bn, net.d1c_scale, net.d1c_relu = conv_bn_relu(net.d1b_conv, 128, pad=1, kernel_size=3, stride=1)
	#net.d1d_conv, net.d1d_bn, net.d1d_scale = conv_bn(net.d1c_conv, 128, pad=1, kernel_size=3, stride=1)
	net.d1e_conv, net.d1e_bn, net.d1e_scale, net.d1e_eltw, net.d1e_relu = add_layer(net.d1a_pool, net.d1c_conv, 128)
	############ d2 ############
	net.d2a_pool = max_pool(net.d1e_relu, pad=0, kernel_size=2, stride=2)
	net.d2b_conv, net.d2b_bn, net.d2b_scale, net.d2b_relu = conv_bn_relu(net.d2a_pool, 256, pad=1, kernel_size=3, stride=1)
	net.d2c_conv, net.d2c_bn, net.d2c_scale, net.d2c_relu = conv_bn_relu(net.d2b_conv, 256, pad=1, kernel_size=3, stride=1)
	net.d2d_conv, net.d2d_bn, net.d2d_scale = conv_bn(net.d2c_conv, 256, pad=1, kernel_size=3, stride=1)
	net.d2e_conv, net.d2e_bn, net.d2e_scale, net.d2e_eltw, net.d2e_relu = add_layer(net.d2a_pool, net.d2d_conv, 256)
	############ d3 ############
	net.d3a_pool = max_pool(net.d2e_relu, pad=0, kernel_size=2, stride=2)
	net.d3b_conv, net.d3b_bn, net.d3b_scale, net.d3b_relu = conv_bn_relu(net.d3a_pool, 512, pad=1, kernel_size=3, stride=1)
	net.d3c_conv, net.d3c_bn, net.d3c_scale, net.d3c_relu = conv_bn_relu(net.d3b_conv, 512, pad=1, kernel_size=3, stride=1)
	net.d3d_conv, net.d3d_bn, net.d3d_scale = conv_bn(net.d3c_conv, 512, pad=1, kernel_size=3, stride=1)
	net.d3e_conv, net.d3e_bn, net.d3e_scale, net.d3e_eltw, net.d3e_relu = add_layer(net.d3a_pool, net.d3d_conv, 512)
	############ d4 ############
	net.d4a_pool = max_pool(net.d3e_relu, pad=0, kernel_size=2, stride=2)
	net.d4b_conv, net.d4b_bn, net.d4b_scale, net.d4b_relu = conv_bn_relu(net.d4a_pool, 1024, pad=1, kernel_size=3, stride=1)
	net.d4c_conv, net.d4c_bn, net.d4c_scale, net.d4c_relu = conv_bn_relu(net.d4b_conv, 1024, pad=1, kernel_size=3, stride=1)
	net.d4d_conv, net.d4d_bn, net.d4d_scale = conv_bn(net.d4c_conv, 1024, pad=1, kernel_size=3, stride=1)
	net.d4e_conv, net.d4e_bn, net.d4e_scale, net.d4e_eltw, net.d4e_relu = add_layer(net.d4a_pool, net.d4d_conv, 1024)
	############ u3 ############
	### a ### First Deconvolution
	net.u3a_dconv, net.u3a_bn, net.u3a_scale, net.u3a_relu = deconv_bn_relu(net.d4e_relu, 512, pad=0, kernel_size=2, stride=2)
	net.u3b_concat = L.Concat(net.u3a_dconv, net.d3e_relu, axis=1)
	net.u3c_conv, net.u3c_bn, net.u3c_scale, net.u3c_relu = conv_bn_relu(net.u3b_concat, 512, pad=1, kernel_size=3, stride=1)
	net.u3d_conv, net.u3d_bn, net.u3d_scale, net.u3d_relu = conv_bn_relu(net.u3c_conv, 512, pad=1, kernel_size=3, stride=1)
	net.u3e_conv, net.u3e_bn, net.u3e_scale = conv_bn(net.u3d_conv, 512, pad=1, kernel_size=3, stride=1)
	net.u3f_conv, net.u3f_bn, net.u3f_scale, net.u3f_eltw, net.u3f_relu = add_layer(net.u3b_concat, net.u3e_conv, 512)
	############ u2 ############
	### a ### Second Deconvolution
	net.u2a_dconv, net.u2a_bn, net.u2a_scale, net.u2a_relu = deconv_bn_relu(net.u3f_relu, 256, pad=0, kernel_size=2, stride=2)
	net.u2b_concat = L.Concat(net.u2a_dconv, net.d2e_relu, axis=1)
	net.u2c_conv, net.u2c_bn, net.u2c_scale, net.u2c_relu = conv_bn_relu(net.u2b_concat, 256, pad=1, kernel_size=3, stride=1)
	net.u2d_conv, net.u2d_bn, net.u2d_scale, net.u2d_relu = conv_bn_relu(net.u2c_conv, 256, pad=1, kernel_size=3, stride=1)
	net.u2e_conv, net.u2e_bn, net.u2e_scale = conv_bn(net.u2d_conv, 256, pad=1, kernel_size=3, stride=1)
	net.u2f_conv, net.u2f_bn, net.u2f_scale, net.u2f_eltw, net.u2f_relu = add_layer(net.u2b_concat, net.u2e_conv, 256)
	############ u1 ############
	### a ### Third Deconvolution
	net.u1a_dconv, net.u1a_bn, net.u1a_scale, net.u1a_relu = deconv_bn_relu(net.u2f_relu, 128, pad=0, kernel_size=2, stride=2)
	net.u1b_concat = L.Concat(net.u1a_dconv, net.d1e_relu, axis=1)
	net.u1c_conv, net.u1c_bn, net.u1c_scale, net.u1c_relu = conv_bn_relu(net.u1b_concat, 128, pad=1, kernel_size=3, stride=1)
	# net.u1d_conv, net.u1d_bn, net.u1d_scale, net.u1d_relu = conv_bn_relu(net.u1c_conv, 128, pad=1, kernel_size=3, stride=1)
	net.u1e_conv, net.u1e_bn, net.u1e_scale = conv_bn(net.u1c_conv, 128, pad=1, kernel_size=3, stride=1)
	net.u1f_conv, net.u1f_bn, net.u1f_scale, net.u1f_eltw, net.u1f_relu = add_layer(net.u1b_concat, net.u1e_conv, 128)
	############ u0 ############
	### a ### Fourth Deconvolution
	net.u0a_dconv, net.u0a_bn, net.u0a_scale, net.u0a_relu = deconv_bn_relu(net.u1f_relu, 64, pad=0, kernel_size=2, stride=2)
	net.u0b_concat = L.Concat(net.u0a_dconv, net.d0c_conv, axis=1)
	net.u0c_conv, net.u0c_bn, net.u0c_scale, net.u0c_relu = conv_bn_relu(net.u0b_concat, 64, pad=1, kernel_size=3, stride=1)
	net.u0d_conv, net.u0d_bn, net.u0d_scale = conv_bn(net.u0c_conv, 64, pad=1, kernel_size=3, stride=1)
	net.u0e_conv, net.u0e_bn, net.u0e_scale, net.u0e_eltw, net.u0e_relu = add_layer(net.u0b_concat, net.u0d_conv, 64)
	############ score ###########
	net.score3 = L.Convolution(net.u3f_relu,
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
	net.score2 = L.Convolution(net.u2f_relu,
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
	net.score1 = L.Convolution(net.u1f_relu,
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
	net.score = L.Convolution(net.u0e_relu,
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
	return net.to_proto()


def uvnet_2d_bn_original_weigted(dim_data, dim_label, num_class, phase='train'):
	net = caffe.NetSpec()
	############ d0 ############
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if phase == "train":
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		net.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))	
	net.d0b_conv, net.d0b_bn, net.d0b_scale, net.d0b_relu = conv_bn_relu(net.data, 64, pad=1, kernel_size=3, stride=1)
	net.d0c_conv, net.d0c_bn, net.d0c_scale, net.d0c_relu = conv_bn_relu(net.d0b_conv, 64, pad=1, kernel_size=3, stride=1)
	############ d1 ############
	net.d1a_pool = max_pool(net.d0c_conv, pad=0, kernel_size=2, stride=2)
	net.d1b_conv, net.d1b_bn, net.d1b_scale, net.d1b_relu = conv_bn_relu(net.d1a_pool, 128, pad=1, kernel_size=3, stride=1)
	# net.d1c_conv, net.d1c_bn, net.d1c_scale, net.d1c_relu = conv_bn_relu(net.d1b_conv, 128, pad=1, kernel_size=3, stride=1)
	net.d1d_conv, net.d1d_bn, net.d1d_scale = conv_bn(net.d1b_relu, 128, pad=1, kernel_size=3, stride=1)
	net.d1e_conv, net.d1e_bn, net.d1e_scale, net.d1e_eltw, net.d1e_relu = add_layer(net.d1a_pool, net.d1d_conv, 128)
	############ d2 ############
	net.d2a_pool = max_pool(net.d1e_relu, pad=0, kernel_size=2, stride=2)
	net.d2b_conv, net.d2b_bn, net.d2b_scale, net.d2b_relu = conv_bn_relu(net.d2a_pool, 256, pad=1, kernel_size=3, stride=1)
	net.d2c_conv, net.d2c_bn, net.d2c_scale, net.d2c_relu = conv_bn_relu(net.d2b_conv, 256, pad=1, kernel_size=3, stride=1)
	net.d2d_conv, net.d2d_bn, net.d2d_scale = conv_bn(net.d2c_conv, 256, pad=1, kernel_size=3, stride=1)
	net.d2e_conv, net.d2e_bn, net.d2e_scale, net.d2e_eltw, net.d2e_relu = add_layer(net.d2a_pool, net.d2d_conv, 256)
	############ d3 ############
	net.d3a_pool = max_pool(net.d2e_relu, pad=0, kernel_size=2, stride=2)
	net.d3b_conv, net.d3b_bn, net.d3b_scale, net.d3b_relu = conv_bn_relu(net.d3a_pool, 512, pad=1, kernel_size=3, stride=1)
	net.d3c_conv, net.d3c_bn, net.d3c_scale, net.d3c_relu = conv_bn_relu(net.d3b_conv, 512, pad=1, kernel_size=3, stride=1)
	net.d3d_conv, net.d3d_bn, net.d3d_scale = conv_bn(net.d3c_conv, 512, pad=1, kernel_size=3, stride=1)
	net.d3e_conv, net.d3e_bn, net.d3e_scale, net.d3e_eltw, net.d3e_relu = add_layer(net.d3a_pool, net.d3d_conv, 512)
	############ d4 ############
	net.d4a_pool = max_pool(net.d3e_relu, pad=0, kernel_size=2, stride=2)
	net.d4b_conv, net.d4b_bn, net.d4b_scale, net.d4b_relu = conv_bn_relu(net.d4a_pool, 1024, pad=1, kernel_size=3, stride=1)
	net.d4c_conv, net.d4c_bn, net.d4c_scale, net.d4c_relu = conv_bn_relu(net.d4b_conv, 1024, pad=1, kernel_size=3, stride=1)
	net.d4d_conv, net.d4d_bn, net.d4d_scale = conv_bn(net.d4c_conv, 1024, pad=1, kernel_size=3, stride=1)
	net.d4e_conv, net.d4e_bn, net.d4e_scale, net.d4e_eltw, net.d4e_relu = add_layer(net.d4a_pool, net.d4d_conv, 1024)
	############ u3 ############
	### a ### First Deconvolution
	net.u3a_dconv, net.u3a_bn, net.u3a_scale, net.u3a_relu = deconv_bn_relu(net.d4e_relu, 512, pad=0, kernel_size=2, stride=2)
	net.u3b_concat = L.Concat(net.u3a_dconv, net.d3e_relu, axis=1)
	net.u3c_conv, net.u3c_bn, net.u3c_scale, net.u3c_relu = conv_bn_relu(net.u3b_concat, 512, pad=1, kernel_size=3, stride=1)
	net.u3d_conv, net.u3d_bn, net.u3d_scale, net.u3d_relu = conv_bn_relu(net.u3c_conv, 512, pad=1, kernel_size=3, stride=1)
	net.u3e_conv, net.u3e_bn, net.u3e_scale = conv_bn(net.u3d_conv, 512, pad=1, kernel_size=3, stride=1)
	net.u3f_conv, net.u3f_bn, net.u3f_scale, net.u3f_eltw, net.u3f_relu = add_layer(net.u3b_concat, net.u3e_conv, 512)
	############ u2 ############
	### a ### Second Deconvolution
	net.u2a_dconv, net.u2a_bn, net.u2a_scale, net.u2a_relu = deconv_bn_relu(net.u3f_relu, 256, pad=0, kernel_size=2, stride=2)
	net.u2b_concat = L.Concat(net.u2a_dconv, net.d2e_relu, axis=1)
	net.u2c_conv, net.u2c_bn, net.u2c_scale, net.u2c_relu = conv_bn_relu(net.u2b_concat, 256, pad=1, kernel_size=3, stride=1)
	net.u2d_conv, net.u2d_bn, net.u2d_scale, net.u2d_relu = conv_bn_relu(net.u2c_conv, 256, pad=1, kernel_size=3, stride=1)
	net.u2e_conv, net.u2e_bn, net.u2e_scale = conv_bn(net.u2d_conv, 256, pad=1, kernel_size=3, stride=1)
	net.u2f_conv, net.u2f_bn, net.u2f_scale, net.u2f_eltw, net.u2f_relu = add_layer(net.u2b_concat, net.u2e_conv, 256)
	############ u1 ############
	### a ### Third Deconvolution
	net.u1a_dconv, net.u1a_bn, net.u1a_scale, net.u1a_relu = deconv_bn_relu(net.u2f_relu, 128, pad=0, kernel_size=2, stride=2)
	net.u1b_concat = L.Concat(net.u1a_dconv, net.d1e_relu, axis=1)
	net.u1c_conv, net.u1c_bn, net.u1c_scale, net.u1c_relu = conv_bn_relu(net.u1b_concat, 128, pad=1, kernel_size=3, stride=1)
	# net.u1d_conv, net.u1d_bn, net.u1d_scale, net.u1d_relu = conv_bn_relu(net.u1c_conv, 128, pad=1, kernel_size=3, stride=1)
	net.u1e_conv, net.u1e_bn, net.u1e_scale = conv_bn(net.u1c_conv, 128, pad=1, kernel_size=3, stride=1)
	net.u1f_conv, net.u1f_bn, net.u1f_scale, net.u1f_eltw, net.u1f_relu = add_layer(net.u1b_concat, net.u1e_conv, 128)
	############ u0 ############
	### a ### Fourth Deconvolution
	net.u0a_dconv, net.u0a_bn, net.u0a_scale, net.u0a_relu = deconv_bn_relu(net.u1f_relu, 64, pad=0, kernel_size=2, stride=2)
	net.u0b_concat = L.Concat(net.u0a_dconv, net.d0c_conv, axis=1)
	net.u0c_conv, net.u0c_bn, net.u0c_scale, net.u0c_relu = conv_bn_relu(net.u0b_concat, 64, pad=1, kernel_size=3, stride=1)
	net.u0d_conv, net.u0d_bn, net.u0d_scale = conv_bn(net.u0c_conv, 64, pad=1, kernel_size=3, stride=1)
	net.u0e_conv, net.u0e_bn, net.u0e_scale, net.u0e_eltw, net.u0e_relu = add_layer(net.u0b_concat, net.u0d_conv, 64)
	############ score ###########
	net.score3 = L.Convolution(net.u3f_relu,
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
	net.score2 = L.Convolution(net.u2f_relu,
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
	net.score1 = L.Convolution(net.u1f_relu,
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
	net.score = L.Convolution(net.u0e_relu,
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
	return net.to_proto()

def uvnet_2d_bn_modified_weigted(dim_data, dim_label, num_class, phase='train'):
	net = caffe.NetSpec()
	############ d0 ############
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if phase == "train":
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		net.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))	
	net.d0b_conv, net.d0b_bn, net.d0b_scale, net.d0b_relu = conv_bn_relu(net.data, 64, pad=1, kernel_size=3, stride=1)
	net.d0c_conv, net.d0c_bn, net.d0c_scale, net.d0c_relu = conv_bn_relu(net.d0b_conv, 64, pad=1, kernel_size=3, stride=1)
	############ d1 ############
	net.d1a_pool = max_pool(net.d0c_conv, pad=0, kernel_size=2, stride=2)
	net.d1b_conv, net.d1b_bn, net.d1b_scale, net.d1b_relu = conv_bn_relu(net.d1a_pool, 128, pad=1, kernel_size=3, stride=1)
	net.d1c_conv, net.d1c_bn, net.d1c_scale, net.d1c_relu = conv_bn_relu(net.d1b_conv, 128, pad=1, kernel_size=3, stride=1)
	net.d1d_conv1, net.d1d_conv2, net.d1d_eltw, net.d1d_bn, net.d1d_scale, net.d1d_relu  = conv1_conv2_add_bn_relu(net.d1c_conv, net.d1a_pool, 128, pad=1, kernel_size=3, stride=1)
	############ d2 ############
	net.d2a_pool = max_pool(net.d1d_relu, pad=0, kernel_size=2, stride=2)
	net.d2b_conv, net.d2b_bn, net.d2b_scale, net.d2b_relu = conv_bn_relu(net.d2a_pool, 256, pad=1, kernel_size=3, stride=1)
	net.d2c_conv, net.d2c_bn, net.d2c_scale, net.d2c_relu = conv_bn_relu(net.d2b_conv, 256, pad=1, kernel_size=3, stride=1)
	net.d2d_conv1, net.d2d_conv2, net.d2d_eltw, net.d2d_bn, net.d2d_scale, net.d2d_relu  = conv1_conv2_add_bn_relu(net.d2c_conv, net.d2a_pool, 256, pad=1, kernel_size=3, stride=1)
	############ d3 ############
	net.d3a_pool = max_pool(net.d2d_relu, pad=0, kernel_size=2, stride=2)
	net.d3b_conv, net.d3b_bn, net.d3b_scale, net.d3b_relu = conv_bn_relu(net.d3a_pool, 512, pad=1, kernel_size=3, stride=1)
	net.d3c_conv, net.d3c_bn, net.d3c_scale, net.d3c_relu = conv_bn_relu(net.d3b_conv, 512, pad=1, kernel_size=3, stride=1)
	net.d3d_conv1, net.d3d_conv2, net.d3d_eltw, net.d3d_bn, net.d3d_scale, net.d3d_relu  = conv1_conv2_add_bn_relu(net.d3c_conv, net.d3a_pool, 512, pad=1, kernel_size=3, stride=1)
	############ d4 ############
	net.d4a_pool = max_pool(net.d3d_relu, pad=0, kernel_size=2, stride=2)
	net.d4b_conv, net.d4b_bn, net.d4b_scale, net.d4b_relu = conv_bn_relu(net.d4a_pool, 1024, pad=1, kernel_size=3, stride=1)
	net.d4c_conv, net.d4c_bn, net.d4c_scale, net.d4c_relu = conv_bn_relu(net.d4b_conv, 1024, pad=1, kernel_size=3, stride=1)
	net.d4d_conv1, net.d4d_conv2, net.d4d_eltw, net.d4d_bn, net.d4d_scale, net.d4d_relu  = conv1_conv2_add_bn_relu(net.d4c_conv, net.d4a_pool, 1024, pad=1, kernel_size=3, stride=1)
	############ u3 ############
	### a ### First Deconvolution
	net.u3a_dconv, net.u3a_bn, net.u3a_scale, net.u3a_relu = deconv_bn_relu(net.d4d_relu, 512, pad=0, kernel_size=2, stride=2)
	net.u3b_concat = L.Concat(net.u3a_dconv, net.d3d_relu, axis=1)
	net.u3c_conv, net.u3c_bn, net.u3c_scale, net.u3c_relu = conv_bn_relu(net.u3b_concat, 512, pad=1, kernel_size=3, stride=1)
	net.u3d_conv, net.u3d_bn, net.u3d_scale, net.u3d_relu = conv_bn_relu(net.u3c_conv, 512, pad=1, kernel_size=3, stride=1)
	net.u3e_conv1, net.u3e_conv2, net.u3e_eltw, net.u3e_bn, net.u3e_scale, net.u3e_relu  = conv1_conv2_add_bn_relu(net.u3d_conv, net.u3b_concat, 512, pad=1, kernel_size=3, stride=1)
	############ u2 ############
	### a ### Second Deconvolution
	net.u2a_dconv, net.u2a_bn, net.u2a_scale, net.u2a_relu = deconv_bn_relu(net.u3e_relu, 256, pad=0, kernel_size=2, stride=2)
	net.u2b_concat = L.Concat(net.u2a_dconv, net.d2d_relu, axis=1)
	net.u2c_conv, net.u2c_bn, net.u2c_scale, net.u2c_relu = conv_bn_relu(net.u2b_concat, 256, pad=1, kernel_size=3, stride=1)
	net.u2d_conv, net.u2d_bn, net.u2d_scale, net.u2d_relu = conv_bn_relu(net.u2c_conv, 256, pad=1, kernel_size=3, stride=1)
	net.u2e_conv1, net.u2e_conv2, net.u2e_eltw, net.u2e_bn, net.u2e_scale, net.u2e_relu  = conv1_conv2_add_bn_relu(net.u2d_conv, net.u2b_concat, 256, pad=1, kernel_size=3, stride=1)
	############ u1 ############
	### a ### Third Deconvolution
	net.u1a_dconv, net.u1a_bn, net.u1a_scale, net.u1a_relu = deconv_bn_relu(net.u2e_relu, 128, pad=0, kernel_size=2, stride=2)
	net.u1b_concat = L.Concat(net.u1a_dconv, net.d1d_relu, axis=1)
	net.u1c_conv, net.u1c_bn, net.u1c_scale, net.u1c_relu = conv_bn_relu(net.u1b_concat, 128, pad=1, kernel_size=3, stride=1)
	net.u1d_conv, net.u1d_bn, net.u1d_scale, net.u1d_relu = conv_bn_relu(net.u1c_conv, 128, pad=1, kernel_size=3, stride=1)
	net.u1e_conv1, net.u1e_conv2, net.u1e_eltw, net.u1e_bn, net.u1e_scale, net.u1e_relu  = conv1_conv2_add_bn_relu(net.u1d_conv, net.u1b_concat, 128, pad=1, kernel_size=3, stride=1)
	############ u0 ############
	### a ### Fourth Deconvolution
	net.u0a_dconv, net.u0a_bn, net.u0a_scale, net.u0a_relu = deconv_bn_relu(net.u1e_relu, 64, pad=0, kernel_size=2, stride=2)
	net.u0b_concat = L.Concat(net.u0a_dconv, net.d0c_conv, axis=1)
	net.u0c_conv, net.u0c_bn, net.u0c_scale, net.u0c_relu = conv_bn_relu(net.u0b_concat, 64, pad=1, kernel_size=3, stride=1)
	net.u0d_conv1, net.u0d_conv2, net.u0d_eltw, net.u0d_bn, net.u0d_scale, net.u0d_relu  = conv1_conv2_add_bn_relu(net.u0c_conv, net.u0b_concat, 64, pad=1, kernel_size=3, stride=1)
	############ score ###########
	net.score3 = L.Convolution(net.u3e_relu,
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
	net.score2 = L.Convolution(net.u2e_relu,
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
	net.score1 = L.Convolution(net.u1e_relu,
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
	net.score = L.Convolution(net.u0d_relu,
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
	return net.to_proto()


def make_uvnet(net, dim_data, dim_label, num_class, prototxt_train, prototxt_test):
	# register net
	__nets = ['uvnet_2d_bn', 'uvnet_2d_bn_weigted', 'uvnet_2d_bn_original_weigted', 'uvnet_2d_bn_modified_weigted']
	assert net in __nets, 'Unknown net: {}'.format(net)
	global use_global_stats, engine, ignore_label
	engine = 2
	ignore_label = 255
	if net == 'uvnet_2d_bn':
		use_global_stats = 0
		train_net = uvnet_2d_bn(dim_data, dim_label, num_class, phase='train')
		use_global_stats = 1
		test_net = uvnet_2d_bn(dim_data, dim_label, num_class, phase='test')

	if net == 'uvnet_2d_bn_weigted':
		use_global_stats = 0
		train_net = uvnet_2d_bn_weigted(dim_data, dim_label, num_class, phase='train')
		use_global_stats = 1
		test_net = uvnet_2d_bn_weigted(dim_data, dim_label, num_class, phase='test')

	if net == 'uvnet_2d_bn_original_weigted':
		use_global_stats = 0
		train_net = uvnet_2d_bn_original_weigted(dim_data, dim_label, num_class, phase='train')
		use_global_stats = 1
		test_net = uvnet_2d_bn_original_weigted(dim_data, dim_label, num_class, phase='test')

	if net == 'uvnet_2d_bn_modified_weigted':
		use_global_stats = 0
		train_net = uvnet_2d_bn_modified_weigted(dim_data, dim_label, num_class, phase='train')
		use_global_stats = 1
		test_net = uvnet_2d_bn_modified_weigted(dim_data, dim_label, num_class, phase='test')

	with open(prototxt_train, 'w') as f:
		f.write(str(train_net))
	with open(prototxt_test, 'w') as f:
		f.write(str(test_net))
