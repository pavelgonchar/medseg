import _init_paths
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.coord_map import crop # crop(net.upscore, net.data) automatically calculate the axis and offset
import os
def bn_relu_conv(bottom, num_output, pad=0, kernel_size=3, stride=1, dropout=0.5):
	bn = L.BatchNorm(bottom, in_place=False, use_global_stats=use_global_stats, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	scale = L.Scale(bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	relu = L.ReLU(scale, in_place=True, engine=engine)
	conv = L.Convolution(relu, num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],		
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=engine)
	drop = L.Dropout(conv, dropout_ratio = dropout)
	return bn, scale, relu, conv, drop

def add_layer(bottom, num_output, dropout):
	bn, scale, relu, conv, drop = bn_relu_conv(bottom, num_output=num_output, pad=1, kernel_size=3, stride=1, dropout=dropout)
	concate = L.Concat(bottom, drop, axis=1)
	return bn, scale, relu, conv, drop, concate

def bn_relu_deconv(bottom, num_output, pad=0, kernel_size=3, stride=1):
	bn = L.BatchNorm(bottom, in_place=False, use_global_stats=use_global_stats, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	scale = L.Scale(bn, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	relu = L.ReLU(scale, in_place=True, engine=engine)
	deconv = L.Deconvolution(relu, param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=num_output,  pad=pad, kernel_size=kernel_size, stride=stride,
			weight_filler=dict(type='msra'), bias_term=0,
			engine=engine))
	return bn, scale, relu, deconv

def transition_down(bottom, num_output, dropout):
	bn, scale, relu, conv, drop = bn_relu_conv(bottom, num_output=num_output, pad=0, kernel_size=1, stride=1, dropout=dropout)
	pooling = L.Pooling(drop, pool=P.Pooling.MAX, pad=0, kernel_size=2, stride=2, engine=engine)
	return bn, scale, relu, conv, drop, pooling



############ ############
'''
Change the line below to experiment with different setting
depth -- must be 3n+4
first_output -- # channels before entering the first dense block, set it to be comparable to growth_rate
growth_rate -- growth rate
dropout -- set to 0 to disable dropout, non-zero nuber to set dropout rate
'''
def densenet(dim_data, num_class, phase='train', depth=40, first_output=64, growth_rate=16, dropout=0.2):
	net = caffe.NetSpec()
	### input
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if  phase == 'train':
		dim_label = dim_data
		dim_label[1] = 1
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
	###
	############ d0 ############
	nchannels = first_output
	net.d0a_conv = L.Convolution(net.data, num_output=nchannels, pad=1, kernel_size=3, stride=1,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=engine)
	### dense block
	net.d0b_bn, net.d0b_scale, net.d0b_relu, net.d0b_conv, net.d0b_drop, net.d0b_concate = add_layer(net.d0a_conv, growth_rate, dropout)
	nchannels += growth_rate
	net.d0c_bn, net.d0c_scale, net.d0c_relu, net.d0c_conv, net.d0c_drop, net.d0c_concate = add_layer(net.d0b_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.d0d_bn, net.d0d_scale, net.d0d_relu, net.d0d_conv, net.d0d_drop, net.d0d_concate = add_layer(net.d0c_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.d0e_bn, net.d0e_scale, net.d0e_relu, net.d0e_conv, net.d0e_drop, net.d0e_concate = add_layer(net.d0d_concate, growth_rate, dropout)
	nchannels += growth_rate

	############ d1 ############
	### transition 1
	net.d1a_bn, net.d1a_scale, net.d1a_relu, net.d1a_conv, net.d1a_drop, net.d1a_pooling = transition_down(net.d0e_concate, nchannels, dropout)
	### dense block
	net.d1b_bn, net.d1b_scale, net.d1b_relu, net.d1b_conv, net.d1b_drop, net.d1b_concate = add_layer(net.d1a_pooling, growth_rate, dropout)
	nchannels += growth_rate
	net.d1c_bn, net.d1c_scale, net.d1c_relu, net.d1c_conv, net.d1c_drop, net.d1c_concate = add_layer(net.d1b_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.d1d_bn, net.d1d_scale, net.d1d_relu, net.d1d_conv, net.d1d_drop, net.d1d_concate = add_layer(net.d1c_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.d1e_bn, net.d1e_scale, net.d1e_relu, net.d1e_conv, net.d1e_drop, net.d1e_concate = add_layer(net.d1d_concate, growth_rate, dropout)
	nchannels += growth_rate

	############ d2 ############
	### transition 2
	net.d2a_bn, net.d2a_scale, net.d2a_relu, net.d2a_conv, net.d2a_drop, net.d2a_pooling = transition_down(net.d1e_concate, nchannels, dropout)
	### dense block
	net.d2b_bn, net.d2b_scale, net.d2b_relu, net.d2b_conv, net.d2b_drop, net.d2b_concate = add_layer(net.d2a_pooling, growth_rate, dropout)
	nchannels += growth_rate
	net.d2c_bn, net.d2c_scale, net.d2c_relu, net.d2c_conv, net.d2c_drop, net.d2c_concate = add_layer(net.d2b_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.d2d_bn, net.d2d_scale, net.d2d_relu, net.d2d_conv, net.d2d_drop, net.d2d_concate = add_layer(net.d2c_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.d2e_bn, net.d2e_scale, net.d2e_relu, net.d2e_conv, net.d2e_drop, net.d2e_concate = add_layer(net.d2d_concate, growth_rate, dropout)
	nchannels += growth_rate

	############ d3 ############
	### transition 3
	net.d3a_bn, net.d3a_scale, net.d3a_relu, net.d3a_conv, net.d3a_drop, net.d3a_pooling = transition_down(net.d2e_concate, nchannels, dropout)
	### dense block
	net.d3b_bn, net.d3b_scale, net.d3b_relu, net.d3b_conv, net.d3b_drop, net.d3b_concate = add_layer(net.d3a_pooling, growth_rate, dropout)
	nchannels += growth_rate
	net.d3c_bn, net.d3c_scale, net.d3c_relu, net.d3c_conv, net.d3c_drop, net.d3c_concate = add_layer(net.d3b_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.d3d_bn, net.d3d_scale, net.d3d_relu, net.d3d_conv, net.d3d_drop, net.d3d_concate = add_layer(net.d3c_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.d3e_bn, net.d3e_scale, net.d3e_relu, net.d3e_conv, net.d3e_drop, net.d3e_concate = add_layer(net.d3d_concate, growth_rate, dropout)
	nchannels += growth_rate

	############ d4 ############
	### transition 4
	net.d4a_bn, net.d4a_scale, net.d4a_relu, net.d4a_conv, net.d4a_drop, net.d4a_pooling = transition_down(net.d3e_concate, nchannels, dropout)
	### dense block
	net.d4b_bn, net.d4b_scale, net.d4b_relu, net.d4b_conv, net.d4b_drop, net.d4b_concate = add_layer(net.d4a_pooling, growth_rate, dropout)
	nchannels += growth_rate
	net.d4c_bn, net.d4c_scale, net.d4c_relu, net.d4c_conv, net.d4c_drop, net.d4c_concate = add_layer(net.d4b_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.d4d_bn, net.d4d_scale, net.d4d_relu, net.d4d_conv, net.d4d_drop, net.d4d_concate = add_layer(net.d4c_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.d4e_bn, net.d4e_scale, net.d4e_relu, net.d4e_conv, net.d4e_drop, net.d4e_concate = add_layer(net.d4d_concate, growth_rate, dropout)
	#nchannels += growth_rate

	############ u3 ############
	### a ### First Deconvolution, transition_up
	net.u3a_bn, net.u3a_scale, net.u3a_relu, net.u3a_dconv = bn_relu_deconv(net.d4e_concate, nchannels, pad=0, kernel_size=2, stride=2)
	### Concat
	net.u3a_concat = L.Concat(net.u3a_dconv, net.d3e_concate, axis=1)
	### dense block
	net.u3b_bn, net.u3b_scale, net.u3b_relu, net.u3b_conv, net.u3b_drop, net.u3b_concate = add_layer(net.u3a_concat, growth_rate, dropout)
	nchannels += growth_rate
	net.u3c_bn, net.u3c_scale, net.u3c_relu, net.u3c_conv, net.u3c_drop, net.u3c_concate = add_layer(net.u3b_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.u3d_bn, net.u3d_scale, net.u3d_relu, net.u3d_conv, net.u3d_drop, net.u3d_concate = add_layer(net.u3c_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.u3e_bn, net.u3e_scale, net.u3e_relu, net.u3e_conv, net.u3e_drop, net.u3e_concate = add_layer(net.u3d_concate, growth_rate, dropout)
	#nchannels += growth_rate

	############ u2 ############
	### a ### Second Deconvolution, transition_up
	net.u2a_bn, net.u2a_scale, net.u2a_relu, net.u2a_dconv = bn_relu_deconv(net.u3e_concate, nchannels, pad=0, kernel_size=2, stride=2)
	### Concat
	net.u2a_concat = L.Concat(net.u2a_dconv, net.d2e_concate, axis=1)
	### dense block
	net.u2b_bn, net.u2b_scale, net.u2b_relu, net.u2b_conv, net.u2b_drop, net.u2b_concate = add_layer(net.u2a_concat, growth_rate, dropout)
	nchannels += growth_rate
	net.u2c_bn, net.u2c_scale, net.u2c_relu, net.u2c_conv, net.u2c_drop, net.u2c_concate = add_layer(net.u2b_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.u2d_bn, net.u2d_scale, net.u2d_relu, net.u2d_conv, net.u2d_drop, net.u2d_concate = add_layer(net.u2c_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.u2e_bn, net.u2e_scale, net.u2e_relu, net.u2e_conv, net.u2e_drop, net.u2e_concate = add_layer(net.u2d_concate, growth_rate, dropout)
	#nchannels += growth_rate

	############ u1 ############
	### a ### Thrid Deconvolution, transition_up
	net.u1a_bn, net.u1a_scale, net.u1a_relu, net.u1a_dconv = bn_relu_deconv(net.u2e_concate, nchannels, pad=0, kernel_size=2, stride=2)
	### Concat
	net.u1a_concat = L.Concat(net.u1a_dconv, net.d1e_concate, axis=1)
	### dense block
	net.u1b_bn, net.u1b_scale, net.u1b_relu, net.u1b_conv, net.u1b_drop, net.u1b_concate = add_layer(net.u1a_concat, growth_rate, dropout)
	nchannels += growth_rate
	net.u1c_bn, net.u1c_scale, net.u1c_relu, net.u1c_conv, net.u1c_drop, net.u1c_concate = add_layer(net.u1b_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.u1d_bn, net.u1d_scale, net.u1d_relu, net.u1d_conv, net.u1d_drop, net.u1d_concate = add_layer(net.u1c_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.u1e_bn, net.u1e_scale, net.u1e_relu, net.u1e_conv, net.u1e_drop, net.u1e_concate = add_layer(net.u1d_concate, growth_rate, dropout)
	#nchannels += growth_rate

	############ u0 ############
	### a ### First Deconvolution, transition_up
	net.u0a_bn, net.u0a_scale, net.u0a_relu, net.u0a_dconv = bn_relu_deconv(net.u1e_concate, nchannels, pad=0, kernel_size=2, stride=2)
	### Concat
	net.u0a_concat = L.Concat(net.u0a_dconv, net.d0e_concate, axis=1)
	### dense block
	net.u0b_bn, net.u0b_scale, net.u0b_relu, net.u0b_conv, net.u0b_drop, net.u0b_concate = add_layer(net.u0a_concat, growth_rate, dropout)
	nchannels += growth_rate
	net.u0c_bn, net.u0c_scale, net.u0c_relu, net.u0c_conv, net.u0c_drop, net.u0c_concate = add_layer(net.u0b_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.u0d_bn, net.u0d_scale, net.u0d_relu, net.u0d_conv, net.u0d_drop, net.u0d_concate = add_layer(net.u0c_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.u0e_bn, net.u0e_scale, net.u0e_relu, net.u0e_conv, net.u0e_drop, net.u0e_concate = add_layer(net.u0d_concate, growth_rate, dropout)
	#nchannels += growth_rate

	############ loss ############
	### loss 3
	net.score3 = L.Convolution(net.u3e_concate, num_output=num_class, pad=0, kernel_size=1, stride=1, 
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore3 = L.Deconvolution(net.score3, param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=num_class, pad=4, kernel_size=16, stride=8,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss3 = L.SoftmaxWithLoss(net.upscore3, net.label,loss_param=dict(ignore_label=ignore_label),
			loss_weight=0.0625)
	### loss 2
	net.score2 = L.Convolution(net.u2e_concate, num_output=num_class, pad=0, kernel_size=1, stride=1,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore2 = L.Deconvolution(net.score2, param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=num_class, pad=2, kernel_size=8, stride=4,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss2 = L.SoftmaxWithLoss(net.upscore2, net.label,loss_param=dict(ignore_label=ignore_label),
			loss_weight=0.125)
	### loss 1
	net.score1 = L.Convolution(net.u1e_concate, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore1 = L.Deconvolution(net.score1, param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=num_class, pad=1, kernel_size=4, stride=2,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss1 = L.SoftmaxWithLoss(net.upscore1, net.label,loss_param=dict(ignore_label=ignore_label),
			loss_weight=0.5)

	### loss 0
	net.score = L.Convolution(net.u0e_concate, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	if phase == "train":
		net.loss = L.SoftmaxWithLoss(net.score, net.label,loss_param=dict(ignore_label=ignore_label),
			loss_weight=1)

	return net.to_proto()



	############ d2 ############

	# for i in xrange(4):
	# 	model = add_layer(net.d0b_conv, growth_rate, dropout)
	# 	nchannels += growth_rate
	# model = transition(model, nchannels, dropout)

	# for i in range(n):
	# 	model = add_layer(model, growth_rate, dropout)
	# 	nchannels += growth_rate
	# model = transition(model, nchannels, dropout)

	# for i in xrange(n):
	# 	model = add_layer(model, growth_rate, dropout)
	# 	nchannels += growth_rate

	# model = L.BatchNorm(model, in_place=False, use_global_stats=use_global_stats, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	# model = L.Scale(model, in_place=True, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	# model = L.ReLU(model, in_place=True, engine=engine)

	# model = L.Pooling(model, pool=P.Pooling.AVE, global_pooling=True)
	# model = L.InnerProduct(model, num_output=10, bias_term=True, weight_filler=dict(type='xavier'), bias_filler=dict(type='constant'))
	# loss = L.SoftmaxWithLoss(model, label)
	# accuracy = L.Accuracy(model, label)

	# return to_proto(loss, accuracy)

def densenet_unet(dim_data, dim_label, num_class, phase='train', depth=40, first_output=64, growth_rate=16, dropout=0.2):
	net = caffe.NetSpec()
	### input
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if  phase == 'train':
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
	###
	############ d0 ############
	nchannels = first_output
	net.d0a_conv = L.Convolution(net.data, num_output=nchannels, pad=1, kernel_size=3, stride=1,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=engine)
	### dense block
	net.d0b_bn, net.d0b_scale, net.d0b_relu, net.d0b_conv, net.d0b_drop, net.d0b_concate = add_layer(net.d0a_conv, growth_rate, dropout)
	nchannels += growth_rate
	net.d0c_bn, net.d0c_scale, net.d0c_relu, net.d0c_conv, net.d0c_drop, net.d0c_concate = add_layer(net.d0b_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.d0d_bn, net.d0d_scale, net.d0d_relu, net.d0d_conv, net.d0d_drop, net.d0d_concate = add_layer(net.d0c_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.d0e_bn, net.d0e_scale, net.d0e_relu, net.d0e_conv, net.d0e_drop, net.d0e_concate = add_layer(net.d0d_concate, growth_rate, dropout)
	nchannels += growth_rate

	############ d1 ############
	### transition 1
	net.d1a_bn, net.d1a_scale, net.d1a_relu, net.d1a_conv, net.d1a_drop, net.d1a_pooling = transition_down(net.d0e_concate, nchannels, dropout)
	### dense block
	net.d1b_bn, net.d1b_scale, net.d1b_relu, net.d1b_conv, net.d1b_drop, net.d1b_concate = add_layer(net.d1a_pooling, growth_rate, dropout)
	nchannels += growth_rate
	net.d1c_bn, net.d1c_scale, net.d1c_relu, net.d1c_conv, net.d1c_drop, net.d1c_concate = add_layer(net.d1b_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.d1d_bn, net.d1d_scale, net.d1d_relu, net.d1d_conv, net.d1d_drop, net.d1d_concate = add_layer(net.d1c_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.d1e_bn, net.d1e_scale, net.d1e_relu, net.d1e_conv, net.d1e_drop, net.d1e_concate = add_layer(net.d1d_concate, growth_rate, dropout)
	nchannels += growth_rate

	############ d2 ############
	### transition 2
	net.d2a_bn, net.d2a_scale, net.d2a_relu, net.d2a_conv, net.d2a_drop, net.d2a_pooling = transition_down(net.d1e_concate, nchannels, dropout)
	### dense block
	net.d2b_bn, net.d2b_scale, net.d2b_relu, net.d2b_conv, net.d2b_drop, net.d2b_concate = add_layer(net.d2a_pooling, growth_rate, dropout)
	nchannels += growth_rate
	net.d2c_bn, net.d2c_scale, net.d2c_relu, net.d2c_conv, net.d2c_drop, net.d2c_concate = add_layer(net.d2b_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.d2d_bn, net.d2d_scale, net.d2d_relu, net.d2d_conv, net.d2d_drop, net.d2d_concate = add_layer(net.d2c_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.d2e_bn, net.d2e_scale, net.d2e_relu, net.d2e_conv, net.d2e_drop, net.d2e_concate = add_layer(net.d2d_concate, growth_rate, dropout)
	nchannels += growth_rate

	############ d3 ############
	### transition 3
	net.d3a_bn, net.d3a_scale, net.d3a_relu, net.d3a_conv, net.d3a_drop, net.d3a_pooling = transition_down(net.d2e_concate, nchannels, dropout)
	### dense block
	net.d3b_bn, net.d3b_scale, net.d3b_relu, net.d3b_conv, net.d3b_drop, net.d3b_concate = add_layer(net.d3a_pooling, growth_rate, dropout)
	nchannels += growth_rate
	net.d3c_bn, net.d3c_scale, net.d3c_relu, net.d3c_conv, net.d3c_drop, net.d3c_concate = add_layer(net.d3b_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.d3d_bn, net.d3d_scale, net.d3d_relu, net.d3d_conv, net.d3d_drop, net.d3d_concate = add_layer(net.d3c_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.d3e_bn, net.d3e_scale, net.d3e_relu, net.d3e_conv, net.d3e_drop, net.d3e_concate = add_layer(net.d3d_concate, growth_rate, dropout)
	nchannels += growth_rate

	############ d4 ############
	### transition 4
	net.d4a_bn, net.d4a_scale, net.d4a_relu, net.d4a_conv, net.d4a_drop, net.d4a_pooling = transition_down(net.d3e_concate, nchannels, dropout)
	### dense block
	net.d4b_bn, net.d4b_scale, net.d4b_relu, net.d4b_conv, net.d4b_drop, net.d4b_concate = add_layer(net.d4a_pooling, growth_rate, dropout)
	nchannels += growth_rate
	net.d4c_bn, net.d4c_scale, net.d4c_relu, net.d4c_conv, net.d4c_drop, net.d4c_concate = add_layer(net.d4b_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.d4d_bn, net.d4d_scale, net.d4d_relu, net.d4d_conv, net.d4d_drop, net.d4d_concate = add_layer(net.d4c_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.d4e_bn, net.d4e_scale, net.d4e_relu, net.d4e_conv, net.d4e_drop, net.d4e_concate = add_layer(net.d4d_concate, growth_rate, dropout)
	#nchannels += growth_rate

	############ u3 ############
	### a ### First Deconvolution, transition_up
	net.u3a_bn, net.u3a_scale, net.u3a_relu, net.u3a_dconv = bn_relu_deconv(net.d4e_concate, num_output=256, pad=0, kernel_size=2, stride=2)
	### Concat
	net.u3a_concat = L.Concat(net.u3a_dconv, net.d3e_concate, axis=1)
	### dense block
	### c ###
	net.u3b_bn, net.u3b_scale, net.u3b_relu, net.u3b_conv, net.u3b_drop = bn_relu_conv(net.u3a_concat, num_output=256, pad=1, kernel_size=3, stride=1, dropout=dropout)
	net.u3c_bn, net.u3c_scale, net.u3c_relu, net.u3c_conv, net.u3c_drop = bn_relu_conv(net.u3b_drop, num_output=256, pad=1, kernel_size=3, stride=1, dropout=dropout)

	############ u2 ############
	### a ### Second Deconvolution, transition_up
	net.u2a_bn, net.u2a_scale, net.u2a_relu, net.u2a_dconv = bn_relu_deconv(net.u3c_drop, num_output=192, pad=0, kernel_size=2, stride=2)
	### Concat
	net.u2a_concat = L.Concat(net.u2a_dconv, net.d2e_concate, axis=1)
	### dense block
	### c ###
	net.u2b_bn, net.u2b_scale, net.u2b_relu, net.u2b_conv, net.u2b_drop = bn_relu_conv(net.u2a_concat, num_output=192, pad=1, kernel_size=3, stride=1, dropout=dropout)
	net.u2c_bn, net.u2c_scale, net.u2c_relu, net.u2c_conv, net.u2c_drop = bn_relu_conv(net.u2b_drop, num_output=192, pad=1, kernel_size=3, stride=1, dropout=dropout)

	############ u1 ############
	### a ### Third Deconvolution
	net.u1a_bn, net.u1a_scale, net.u1a_relu, net.u1a_dconv = bn_relu_deconv(net.u2c_drop, num_output=128, pad=0, kernel_size=2, stride=2)
	### Concat
	net.u1a_concat = L.Concat(net.u1a_dconv, net.d1e_concate, axis=1)
	### dense block
	### c ###
	net.u1b_bn, net.u1b_scale, net.u1b_relu, net.u1b_conv, net.u1b_drop = bn_relu_conv(net.u1a_concat, num_output=128, pad=1, kernel_size=3, stride=1, dropout=dropout)
	net.u1c_bn, net.u1c_scale, net.u1c_relu, net.u1c_conv, net.u1c_drop = bn_relu_conv(net.u1b_drop, num_output=128, pad=1, kernel_size=3, stride=1, dropout=dropout)

	############ u0 ############
	### a ### Fouth Deconvolution
	net.u0a_bn, net.u0a_scale, net.u0a_relu, net.u0a_dconv = bn_relu_deconv(net.u1c_drop, num_output=64, pad=0, kernel_size=2, stride=2)
	### Concat
	net.u0a_concat = L.Concat(net.u0a_dconv, net.d0e_concate, axis=1)
	### dense block
	### c ###
	net.u0b_bn, net.u0b_scale, net.u0b_relu, net.u0b_conv, net.u0b_drop = bn_relu_conv(net.u0a_concat, num_output=64, pad=1, kernel_size=3, stride=1, dropout=dropout)
	net.u0c_bn, net.u0c_scale, net.u0c_relu, net.u0c_conv, net.u0c_drop = bn_relu_conv(net.u0b_drop, num_output=64, pad=1, kernel_size=3, stride=1, dropout=dropout)

	############ loss ############
	### loss 3
	net.score3 = L.Convolution(net.u3c_drop, num_output=num_class, pad=0, kernel_size=1, stride=1, 
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore3 = L.Deconvolution(net.score3, param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=num_class, pad=4, kernel_size=16, stride=8,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss3 = L.SoftmaxWithLoss(net.upscore3, net.label,loss_param=dict(ignore_label=ignore_label),
			loss_weight=0.0625)
	### loss 2
	net.score2 = L.Convolution(net.u2c_drop, num_output=num_class, pad=0, kernel_size=1, stride=1,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore2 = L.Deconvolution(net.score2, param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=num_class, pad=2, kernel_size=8, stride=4,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss2 = L.SoftmaxWithLoss(net.upscore2, net.label,loss_param=dict(ignore_label=ignore_label),
			loss_weight=0.125)
	### loss 1
	net.score1 = L.Convolution(net.u1c_drop, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore1 = L.Deconvolution(net.score1, param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=num_class, pad=1, kernel_size=4, stride=2,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss1 = L.SoftmaxWithLoss(net.upscore1, net.label,loss_param=dict(ignore_label=ignore_label),
			loss_weight=0.5)

	### loss 0
	net.score = L.Convolution(net.u0c_drop, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	if phase == "train":
		net.loss = L.SoftmaxWithLoss(net.score, net.label,loss_param=dict(ignore_label=ignore_label),
			loss_weight=1)

	return net.to_proto()


def densenet_unet_8s(dim_data, dim_label, num_class, phase='train', depth=40, first_output=64, growth_rate=16, dropout=0.2):
	net = caffe.NetSpec()
	### input
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if  phase == 'train':
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
	###
	############ d0 ############
	nchannels = first_output
	net.d0a_conv = L.Convolution(net.data, num_output=nchannels, pad=1, kernel_size=3, stride=1,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=engine)
	### dense block
	net.d0b_bn, net.d0b_scale, net.d0b_relu, net.d0b_conv, net.d0b_drop, net.d0b_concate = add_layer(net.d0a_conv, growth_rate, dropout)
	nchannels += growth_rate
	net.d0c_bn, net.d0c_scale, net.d0c_relu, net.d0c_conv, net.d0c_drop, net.d0c_concate = add_layer(net.d0b_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.d0d_bn, net.d0d_scale, net.d0d_relu, net.d0d_conv, net.d0d_drop, net.d0d_concate = add_layer(net.d0c_concate, growth_rate, dropout)
	nchannels += growth_rate
	#net.d0e_bn, net.d0e_scale, net.d0e_relu, net.d0e_conv, net.d0e_drop, net.d0e_concate = add_layer(net.d0d_concate, growth_rate, dropout)
	#nchannels += growth_rate

	############ d1 ############
	### transition 1
	net.d1a_bn, net.d1a_scale, net.d1a_relu, net.d1a_conv, net.d1a_drop, net.d1a_pooling = transition_down(net.d0d_concate, nchannels, dropout)
	### dense block
	net.d1b_bn, net.d1b_scale, net.d1b_relu, net.d1b_conv, net.d1b_drop, net.d1b_concate = add_layer(net.d1a_pooling, growth_rate, dropout)
	nchannels += growth_rate
	net.d1c_bn, net.d1c_scale, net.d1c_relu, net.d1c_conv, net.d1c_drop, net.d1c_concate = add_layer(net.d1b_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.d1d_bn, net.d1d_scale, net.d1d_relu, net.d1d_conv, net.d1d_drop, net.d1d_concate = add_layer(net.d1c_concate, growth_rate, dropout)
	nchannels += growth_rate
	#net.d1e_bn, net.d1e_scale, net.d1e_relu, net.d1e_conv, net.d1e_drop, net.d1e_concate = add_layer(net.d1d_concate, growth_rate, dropout)
	#nchannels += growth_rate

	############ d2 ############
	### transition 2
	net.d2a_bn, net.d2a_scale, net.d2a_relu, net.d2a_conv, net.d2a_drop, net.d2a_pooling = transition_down(net.d1d_concate, nchannels, dropout)
	### dense block
	net.d2b_bn, net.d2b_scale, net.d2b_relu, net.d2b_conv, net.d2b_drop, net.d2b_concate = add_layer(net.d2a_pooling, growth_rate, dropout)
	nchannels += growth_rate
	net.d2c_bn, net.d2c_scale, net.d2c_relu, net.d2c_conv, net.d2c_drop, net.d2c_concate = add_layer(net.d2b_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.d2d_bn, net.d2d_scale, net.d2d_relu, net.d2d_conv, net.d2d_drop, net.d2d_concate = add_layer(net.d2c_concate, growth_rate, dropout)
	nchannels += growth_rate
	#net.d2e_bn, net.d2e_scale, net.d2e_relu, net.d2e_conv, net.d2e_drop, net.d2e_concate = add_layer(net.d2d_concate, growth_rate, dropout)
	#nchannels += growth_rate

	############ d3 ############
	### transition 3
	net.d3a_bn, net.d3a_scale, net.d3a_relu, net.d3a_conv, net.d3a_drop, net.d3a_pooling = transition_down(net.d2d_concate, nchannels, dropout)
	### dense block
	net.d3b_bn, net.d3b_scale, net.d3b_relu, net.d3b_conv, net.d3b_drop, net.d3b_concate = add_layer(net.d3a_pooling, growth_rate, dropout)
	nchannels += growth_rate
	net.d3c_bn, net.d3c_scale, net.d3c_relu, net.d3c_conv, net.d3c_drop, net.d3c_concate = add_layer(net.d3b_concate, growth_rate, dropout)
	nchannels += growth_rate
	net.d3d_bn, net.d3d_scale, net.d3d_relu, net.d3d_conv, net.d3d_drop, net.d3d_concate = add_layer(net.d3c_concate, growth_rate, dropout)
	nchannels += growth_rate
	#net.d3e_bn, net.d3e_scale, net.d3e_relu, net.d3e_conv, net.d3e_drop, net.d3e_concate = add_layer(net.d3d_concate, growth_rate, dropout)
	#nchannels += growth_rate

	# ############ d4 ############
	# ### transition 4
	# net.d4a_bn, net.d4a_scale, net.d4a_relu, net.d4a_conv, net.d4a_drop, net.d4a_pooling = transition_down(net.d3e_concate, nchannels, dropout)
	# ### dense block
	# net.d4b_bn, net.d4b_scale, net.d4b_relu, net.d4b_conv, net.d4b_drop, net.d4b_concate = add_layer(net.d4a_pooling, growth_rate, dropout)
	# nchannels += growth_rate
	# net.d4c_bn, net.d4c_scale, net.d4c_relu, net.d4c_conv, net.d4c_drop, net.d4c_concate = add_layer(net.d4b_concate, growth_rate, dropout)
	# nchannels += growth_rate
	# net.d4d_bn, net.d4d_scale, net.d4d_relu, net.d4d_conv, net.d4d_drop, net.d4d_concate = add_layer(net.d4c_concate, growth_rate, dropout)
	# nchannels += growth_rate
	# net.d4e_bn, net.d4e_scale, net.d4e_relu, net.d4e_conv, net.d4e_drop, net.d4e_concate = add_layer(net.d4d_concate, growth_rate, dropout)
	# #nchannels += growth_rate

	# ############ u3 ############
	# ### a ### First Deconvolution, transition_up
	# net.u3a_bn, net.u3a_scale, net.u3a_relu, net.u3a_dconv = bn_relu_deconv(net.d4e_concate, num_output=256, pad=0, kernel_size=2, stride=2)
	# ### Concat
	# net.u3a_concat = L.Concat(net.u3a_dconv, net.d3e_concate, axis=1)
	# ### dense block
	# ### c ###
	# net.u3b_bn, net.u3b_scale, net.u3b_relu, net.u3b_conv, net.u3b_drop = bn_relu_conv(net.u3a_concat, num_output=256, pad=1, kernel_size=3, stride=1, dropout=dropout)
	# net.u3c_bn, net.u3c_scale, net.u3c_relu, net.u3c_conv, net.u3c_drop = bn_relu_conv(net.u3b_drop, num_output=256, pad=1, kernel_size=3, stride=1, dropout=dropout)

	############ u2 ############
	### a ### Second Deconvolution, transition_up
	net.u2a_bn, net.u2a_scale, net.u2a_relu, net.u2a_dconv = bn_relu_deconv(net.d3d_concate, num_output=192, pad=0, kernel_size=2, stride=2)
	### Concat
	net.u2a_concat = L.Concat(net.u2a_dconv, net.d2d_concate, axis=1)
	### dense block
	### c ###
	net.u2b_bn, net.u2b_scale, net.u2b_relu, net.u2b_conv, net.u2b_drop = bn_relu_conv(net.u2a_concat, num_output=192, pad=1, kernel_size=3, stride=1, dropout=dropout)
	net.u2c_bn, net.u2c_scale, net.u2c_relu, net.u2c_conv, net.u2c_drop = bn_relu_conv(net.u2b_drop, num_output=192, pad=1, kernel_size=3, stride=1, dropout=dropout)

	############ u1 ############
	### a ### Third Deconvolution
	net.u1a_bn, net.u1a_scale, net.u1a_relu, net.u1a_dconv = bn_relu_deconv(net.u2c_drop, num_output=128, pad=0, kernel_size=2, stride=2)
	### Concat
	net.u1a_concat = L.Concat(net.u1a_dconv, net.d1d_concate, axis=1)
	### dense block
	### c ###
	net.u1b_bn, net.u1b_scale, net.u1b_relu, net.u1b_conv, net.u1b_drop = bn_relu_conv(net.u1a_concat, num_output=128, pad=1, kernel_size=3, stride=1, dropout=dropout)
	net.u1c_bn, net.u1c_scale, net.u1c_relu, net.u1c_conv, net.u1c_drop = bn_relu_conv(net.u1b_drop, num_output=128, pad=1, kernel_size=3, stride=1, dropout=dropout)

	############ u0 ############
	### a ### Fouth Deconvolution
	net.u0a_bn, net.u0a_scale, net.u0a_relu, net.u0a_dconv = bn_relu_deconv(net.u1c_drop, num_output=64, pad=0, kernel_size=2, stride=2)
	### Concat
	net.u0a_concat = L.Concat(net.u0a_dconv, net.d0d_concate, axis=1)
	### dense block
	### c ###
	net.u0b_bn, net.u0b_scale, net.u0b_relu, net.u0b_conv, net.u0b_drop = bn_relu_conv(net.u0a_concat, num_output=64, pad=1, kernel_size=3, stride=1, dropout=dropout)
	net.u0c_bn, net.u0c_scale, net.u0c_relu, net.u0c_conv, net.u0c_drop = bn_relu_conv(net.u0b_drop, num_output=64, pad=1, kernel_size=3, stride=1, dropout=dropout)

	############ loss ############
	### loss 3
	# net.score3 = L.Convolution(net.u3c_drop, num_output=num_class, pad=0, kernel_size=1, stride=1, 
	# 	param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
	# 	weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	# net.upscore3 = L.Deconvolution(net.score3, param=[dict(lr_mult=1, decay_mult=1)],
	# 	convolution_param=dict(num_output=num_class, pad=4, kernel_size=16, stride=8,
	# 		weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	# if phase == "train":
	# 	net.loss3 = L.SoftmaxWithLoss(net.upscore3, net.label,loss_param=dict(ignore_label=ignore_label),
	# 		loss_weight=0.0625)
	### loss 2
	net.score2 = L.Convolution(net.u2c_drop, num_output=num_class, pad=0, kernel_size=1, stride=1,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore2 = L.Deconvolution(net.score2, param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=num_class, pad=2, kernel_size=8, stride=4,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss2 = L.SoftmaxWithLoss(net.upscore2, net.label,loss_param=dict(ignore_label=ignore_label),
			loss_weight=0.25)
	### loss 1
	net.score1 = L.Convolution(net.u1c_drop, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	net.upscore1 = L.Deconvolution(net.score1, param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=num_class, pad=1, kernel_size=4, stride=2,
			weight_filler=dict(type='bilinear'), bias_term=0, engine=engine))
	if phase == "train":
		net.loss1 = L.SoftmaxWithLoss(net.upscore1, net.label,loss_param=dict(ignore_label=ignore_label),
			loss_weight=0.5)

	### loss 0
	net.score = L.Convolution(net.u0c_drop, param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0), engine=engine)
	if phase == "train":
		net.loss = L.SoftmaxWithLoss(net.score, net.label,loss_param=dict(ignore_label=ignore_label),
			loss_weight=1)
	else:
		net.prob = L.Softmax(net.score, axis=1)

	return net.to_proto()

def make_densenet(net, dim_data, dim_label, num_class, prototxt_train, prototxt_test):
	# register net
	__nets = ['densenet_unet', 'densenet_unet_8s']
	assert net in __nets, 'Unknown net: {}'.format(net)
	global use_global_stats, engine, ignore_label
	engine = 2
	ignore_label = 255
	if net == 'densenet_unet':
		use_global_stats = 0
		train_net = densenet_unet(dim_data, dim_label, num_class, phase='train')
		use_global_stats = 1
		test_net = densenet_unet(dim_data, dim_label, num_class, phase='test')
	if net == 'densenet_unet_8s':
		use_global_stats = 0
		train_net = densenet_unet_8s(dim_data, dim_label, num_class, phase='train')
		use_global_stats = 1
		test_net = densenet_unet_8s(dim_data, dim_label, num_class, phase='test')

	with open(prototxt_train, 'w') as f:
		f.write(str(train_net))
	with open(prototxt_test, 'w') as f:
		f.write(str(test_net))


