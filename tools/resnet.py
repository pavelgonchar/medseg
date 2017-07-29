import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop # crop(net.upscore, net.data) automatically calculate the axis and offset

############ ############
def conv_bn_scale_relu(bottom, num_output=64, kernel_size=3, pad=0, stride=1, bias_term=False):
	conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, pad=pad, stride=stride, bias_term=bias_term, engine=1, weight_filler=dict(type='gaussian', std=0.001))
	conv_bn = L.BatchNorm(conv, in_place=True, batch_norm_param =dict(use_global_stats=use_global_stats))
	conv_scale = L.Scale(conv ,in_place=True, scale_param=dict(bias_term=True))
	conv_relu = L.ReLU(conv, in_place=True, engine=1)

	return conv, conv_bn, conv_scale, conv_relu

def conv_bn_scale(bottom, num_output=64, kernel_size=3, pad=0, stride=1, bias_term=False):
	conv = L.Convolution(bottom, num_output=num_output, kernel_size=kernel_size, pad=pad, stride=stride, bias_term=bias_term, engine=1, weight_filler=dict(type='gaussian', std=0.001))
	conv_bn = L.BatchNorm(conv, in_place=True, batch_norm_param =dict(use_global_stats=use_global_stats))
	conv_scale = L.Scale(conv, in_place=True, scale_param=dict(bias_term=True))

	return conv, conv_bn, conv_scale

def eltwize_relu(bottom1, bottom2):
	residual_eltwise = L.Eltwise(bottom1, bottom2, eltwise_param=dict(operation=1))
	residual_eltwise_relu = L.ReLU(residual_eltwise, in_place=True, engine=1)
	
	return residual_eltwise, residual_eltwise_relu

def residual_branch(bottom, base_output=64, stride=1):
	### branch2a ###
	branch2a, branch2a_bn, branch2a_scale, branch2a_relu =  conv_bn_scale_relu(bottom, num_output=base_output, kernel_size=1, pad=0, stride=1) 
	### branch2b ###
	branch2b, branch2b_bn, branch2b_scale, branch2b_relu =  conv_bn_scale_relu(branch2a, num_output=base_output, kernel_size=3, pad=1, stride=1)  
	### branch2c ###
	branch2c, branch2c_bn, branch2c_scale = conv_bn_scale(branch2b, num_output=4 * base_output, kernel_size=1, pad=0, stride=1)  # 4*base_output x n x n
	### residual ###
	residual, residual_relu = eltwize_relu(bottom, branch2c) 
	### return ###
	return branch2a, branch2a_bn, branch2a_scale, branch2a_relu, \
		branch2b, branch2b_bn, branch2b_scale, branch2b_relu, \
		branch2c, branch2c_bn, branch2c_scale, \
		residual, residual_relu

def residual_branch_shortcut(bottom, base_output=64, stride=1):
	### branch1 ###
	branch1, branch1_bn, branch1_scale = conv_bn_scale(bottom, num_output=4 * base_output, kernel_size=1, pad=0, stride=stride)
	### branch2a ###
	branch2a, branch2a_bn, branch2a_scale, branch2a_relu = conv_bn_scale_relu(bottom, num_output=base_output, kernel_size=1, pad=0, stride=stride)
	### branch2b ###
	branch2b, branch2b_bn, branch2b_scale, branch2b_relu = conv_bn_scale_relu(branch2a, num_output=base_output, kernel_size=3, pad=1, stride=1)
	### branch2c ###
	branch2c, branch2c_bn, branch2c_scale = conv_bn_scale(branch2b, num_output=4 * base_output, kernel_size=1, pad=0, stride=1)
	### residual ###
	residual, residual_relu = eltwize_relu(branch1, branch2c)  # 4*base_output x n x n
	### return ###
	return branch1, branch1_bn, branch1_scale, \
		branch2a, branch2a_bn, branch2a_scale, branch2a_relu, \
		branch2b, branch2b_bn, branch2b_scale, branch2b_relu, \
		branch2c, branch2c_bn, branch2c_scale, \
		residual, residual_relu

branch_shortcut_string = 'n.res(stage)a_branch1, n.bn(stage)a_branch1, n.scale(stage)a_branch1, \
        n.res(stage)a_branch2a, n.bn(stage)a_branch2a, n.scale(stage)a_branch2a, n.res(stage)a_branch2a_relu, \
        n.res(stage)a_branch2b, n.bn(stage)a_branch2b, n.scale(stage)a_branch2b, n.res(stage)a_branch2b_relu, \
        n.res(stage)a_branch2c, n.bn(stage)a_branch2c, n.scale(stage)a_branch2c,\
        n.res(stage)a, n.res(stage)a_relu = residual_branch_shortcut((bottom), stride=(stride), base_output=(num))'

branch_string = 'n.res(stage)b(order)_branch2a, n.bn(stage)b(order)_branch2a, n.scale(stage)b(order)_branch2a, n.res(stage)b(order)_branch2a_relu, \
				 n.res(stage)b(order)_branch2b, n.bn(stage)b(order)_branch2b, n.scale(stage)b(order)_branch2b, n.res(stage)b(order)_branch2b_relu, \
				 n.res(stage)b(order)_branch2c, n.bn(stage)b(order)_branch2c, n.scale(stage)b(order)_branch2c, \
				 n.res(stage)b(order), n.res(stage)b(order)_relu = residual_branch((bottom), base_output=(num))'

def resnet(dim_data, dim_label, num_class, ignore_label=255, phase='TRAIN', stages=(3,4,6,3)):
	"""
	(3, 4,  6, 3) for 50 layers
	(3, 4, 23, 3) for 101 layers
	(3, 8, 26, 3) for 152 layers
	"""
	global use_global_stats

	n = caffe.NetSpec()
	###
	n.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if phase == 'TRAIN':
		n.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		n.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		use_global_stats = False
	else:
		use_global_stats = True
	### resnet ###
	n.conv1, n.conv1_bn, n.conv1_scale, n.conv1_relu = conv_bn_scale_relu(n.data, num_output=64, kernel_size=7, stride=2, pad=3, bias_term=True) 
	n.pool1 = L.Pooling(n.conv1, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 64x56x56]
	###
	for num in xrange(len(stages)): # num = 0, 1, 2, 3
		for i in xrange(stages[num]): # (3, 4,  6, 3) for 50 layers
			if i == 0:
				stage_string = branch_shortcut_string
				bottom_string = ['n.pool1', 'n.res2b{}'.format(stages[0]-1), 'n.res3b{}'.format(stages[1]-1), 'n.res4b{}'.format(stages[2]-1)][num]
			else:
				stage_string = branch_string
				if i == 1:
					bottom_string = 'n.res{}a'.format(num+2)
				else:
					bottom_string = 'n.res{}b{}'.format(num+2, i-1)

			exec (stage_string.replace('(stage)', str(num + 2)).
				replace('(bottom)', bottom_string).
				replace('(num)', str(2 ** num * 64)).
				replace('(order)', str(i)).
				replace('(stride)', str(int(num > 0) + 1)))
	### resnet ###
	exec ('n.pool5 = L.Pooling((bottom), pool=P.Pooling.AVE, global_pooling=True)'.replace('(bottom)', 'n.res5b{}'.format(stages[3] - 1)))
	n.classifier = L.InnerProduct(n.pool5, num_output=num_class)
	if phase == 'TRAIN':
		n.loss = L.WeightedSoftmaxWithLoss(n.classifier, n.label)
	# if phase == 'TEST':
	# 	n.accuracy_top1 = L.Accuracy(n.classifier, n.label, include=dict(phase=1))
	# 	n.accuracy_top5 = L.Accuracy(n.classifier, n.label, include=dict(phase=1),accuracy_param=dict(top_k=5))
	return n.to_proto()

def resnet_50(dim_data, dim_label, num_class, ignore_label=255, phase="TRAIN"):
	"""
	(3, 4,  6, 3) for 50 layers
	(3, 4, 23, 3) for 101 layers
	(3, 8, 26, 3) for 152 layers
	"""
	global use_global_stats
	n = caffe.NetSpec()
	### data ###
	n.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if phase == 'TRAIN':
		n.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		# n.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		use_global_stats = False
	else:
		use_global_stats = True
	### conv1 ###
	n.conv1, n.bn_conv1, n.scale_conv1, n.conv1_relu = conv_bn_scale_relu(n.data, num_output=64, kernel_size=7, pad=3, stride=2, bias_term=True) 
	n.pool1 = L.Pooling(n.conv1, kernel_size=3, stride=2, pool=P.Pooling.MAX)  # 64x56x56]
	### stage 2 ### a ###
	n.res2a_branch1,  n.bn2a_branch1,  n.scale2a_branch1,\
	n.res2a_branch2a, n.bn2a_branch2a, n.scale2a_branch2a, n.res2a_branch2a_relu,\
	n.res2a_branch2b, n.bn2a_branch2b, n.scale2a_branch2b, n.res2a_branch2b_relu,\
	n.res2a_branch2c, n.bn2a_branch2c, n.scale2a_branch2c,\
	n.res2a, n.res2a_relu = residual_branch_shortcut(n.pool1, base_output=64,  stride=1)
	### stage 2 ### b ###
	n.res2b_branch2a, n.bn2b_branch2a, n.scale2b_branch2a, n.res2b_branch2a_relu,\
	n.res2b_branch2b, n.bn2b_branch2b, n.scale2b_branch2b, n.res2b_branch2b_relu,\
	n.res2b_branch2c, n.bn2b_branch2c, n.scale2b_branch2c,\
	n.res2b, n.res2b_relu = residual_branch(n.res2a, base_output=64,  stride=1)
	### stage 2 ### c ###
	n.res2c_branch2a, n.bn2c_branch2a, n.scale2c_branch2a, n.res2c_branch2a_relu,\
	n.res2c_branch2b, n.bn2c_branch2b, n.scale2c_branch2b, n.res2c_branch2b_relu,\
	n.res2c_branch2c, n.bn2c_branch2c, n.scale2c_branch2c,\
	n.res2c, n.res2c_relu = residual_branch(n.res2b, base_output=64,  stride=1)
	### stage 3 ### a ###
	n.res3a_branch1,  n.bn3a_branch1,  n.scale3a_branch1,\
	n.res3a_branch2a, n.bn3a_branch2a, n.scale3a_branch2a, n.res3a_branch2a_relu,\
	n.res3a_branch2b, n.bn3a_branch2b, n.scale3a_branch2b, n.res3a_branch2b_relu,\
	n.res3a_branch2c, n.bn3a_branch2c, n.scale3a_branch2c,\
	n.res3a, n.res3a_relu = residual_branch_shortcut(n.res2c, base_output=128,  stride=2)
	### stage 3 ### b ###
	n.res3b_branch2a, n.bn3b_branch2a, n.scale3b_branch2a, n.res3b_branch2a_relu,\
	n.res3b_branch2b, n.bn3b_branch2b, n.scale3b_branch2b, n.res3b_branch2b_relu,\
	n.res3b_branch2c, n.bn3b_branch2c, n.scale3b_branch2c,\
	n.res3b, n.res3b_relu = residual_branch(n.res3a, base_output=128,  stride=1)
	### stage 3 ### c ###
	n.res3c_branch2a, n.bn3c_branch2a, n.scale3c_branch2a, n.res3c_branch2a_relu,\
	n.res3c_branch2b, n.bn3c_branch2b, n.scale3c_branch2b, n.res3c_branch2b_relu,\
	n.res3c_branch2c, n.bn3c_branch2c, n.scale3c_branch2c,\
	n.res3c, n.res3c_relu = residual_branch(n.res3b, base_output=128,  stride=1)
	### stage 3 ### d ###
	n.res3d_branch2a, n.bn3d_branch2a, n.scale3d_branch2a, n.res3d_branch2a_relu,\
	n.res3d_branch2b, n.bn3d_branch2b, n.scale3d_branch2b, n.res3d_branch2b_relu,\
	n.res3d_branch2c, n.bn3d_branch2c, n.scale3d_branch2c,\
	n.res3d, n.res3d_relu = residual_branch(n.res3c, base_output=128,  stride=1)
	### stage 4 ### a ###
	n.res4a_branch1,  n.bn4a_branch1,  n.scale4a_branch1,\
	n.res4a_branch2a, n.bn4a_branch2a, n.scale4a_branch2a, n.res4a_branch2a_relu,\
	n.res4a_branch2b, n.bn4a_branch2b, n.scale4a_branch2b, n.res4a_branch2b_relu,\
	n.res4a_branch2c, n.bn4a_branch2c, n.scale4a_branch2c,\
	n.res4a, n.res4a_relu = residual_branch_shortcut(n.res3d, base_output=256,  stride=2)
	### stage 4 ### b ###
	n.res4b_branch2a, n.bn4b_branch2a, n.scale4b_branch2a, n.res4b_branch2a_relu,\
	n.res4b_branch2b, n.bn4b_branch2b, n.scale4b_branch2b, n.res4b_branch2b_relu,\
	n.res4b_branch2c, n.bn4b_branch2c, n.scale4b_branch2c,\
	n.res4b, n.res4b_relu = residual_branch(n.res4a, base_output=256,  stride=1)
	### stage 4 ### c ###
	n.res4c_branch2a, n.bn4c_branch2a, n.scale4c_branch2a, n.res4c_branch2a_relu,\
	n.res4c_branch2b, n.bn4c_branch2b, n.scale4c_branch2b, n.res4c_branch2b_relu,\
	n.res4c_branch2c, n.bn4c_branch2c, n.scale4c_branch2c,\
	n.res4c, n.res4c_relu = residual_branch(n.res4b, base_output=256,  stride=1)
	### stage 4 ### d ###
	n.res4d_branch2a, n.bn4d_branch2a, n.scale4d_branch2a, n.res4d_branch2a_relu,\
	n.res4d_branch2b, n.bn4d_branch2b, n.scale4d_branch2b, n.res4d_branch2b_relu,\
	n.res4d_branch2c, n.bn4d_branch2c, n.scale4d_branch2c,\
	n.res4d, n.res4d_relu = residual_branch(n.res4c, base_output=256,  stride=1)
	### stage 4 ### e ###
	n.res4e_branch2a, n.bn4e_branch2a, n.scale4e_branch2a, n.res4e_branch2a_relu,\
	n.res4e_branch2b, n.bn4e_branch2b, n.scale4e_branch2b, n.res4e_branch2b_relu,\
	n.res4e_branch2c, n.bn4e_branch2c, n.scale4e_branch2c,\
	n.res4e, n.res4e_relu = residual_branch(n.res4d, base_output=256,  stride=1)
	### stage 4 ### f ###
	n.res4f_branch2a, n.bn4f_branch2a, n.scale4f_branch2a, n.res4f_branch2a_relu,\
	n.res4f_branch2b, n.bn4f_branch2b, n.scale4f_branch2b, n.res4f_branch2b_relu,\
	n.res4f_branch2c, n.bn4f_branch2c, n.scale4f_branch2c,\
	n.res4f, n.res4f_relu = residual_branch(n.res4e, base_output=256,  stride=1)
	### stage 5 ### a ###
	n.res5a_branch1,  n.bn5a_branch1,  n.scale5a_branch1,\
	n.res5a_branch2a, n.bn5a_branch2a, n.scale5a_branch2a, n.res5a_branch2a_relu,\
	n.res5a_branch2b, n.bn5a_branch2b, n.scale5a_branch2b, n.res5a_branch2b_relu,\
	n.res5a_branch2c, n.bn5a_branch2c, n.scale5a_branch2c,\
	n.res5a, n.res5a_relu = residual_branch_shortcut(n.res4f, base_output=512,  stride=2)
	### stage 5 ### b ###
	n.res5b_branch2a, n.bn5b_branch2a, n.scale5b_branch2a, n.res5b_branch2a_relu,\
	n.res5b_branch2b, n.bn5b_branch2b, n.scale5b_branch2b, n.res5b_branch2b_relu,\
	n.res5b_branch2c, n.bn5b_branch2c, n.scale5b_branch2c,\
	n.res5b, n.res5b_relu = residual_branch(n.res5a, base_output=512,  stride=1)
	### stage 5 ### c ###
	n.res5c_branch2a, n.bn5c_branch2a, n.scale5c_branch2a, n.res5c_branch2a_relu,\
	n.res5c_branch2b, n.bn5c_branch2b, n.scale5c_branch2b, n.res5c_branch2b_relu,\
	n.res5c_branch2c, n.bn5c_branch2c, n.scale5c_branch2c,\
	n.res5c, n.res5c_relu = residual_branch(n.res5b, base_output=512,  stride=1)

	n.pool5 = L.Pooling(n.res5c, pool=P.Pooling.AVE, global_pooling=True)
	n.classifier = L.InnerProduct(n.pool5, num_output=num_class)
	if phase == 'TRAIN':
		n.loss = L.SoftmaxWithLoss(n.classifier, n.label)
	return n.to_proto()


# def conv_bn_scale_relu(bottom, num_output, pad=0, kernel_size=3, stride=1, phase='train'):
# 	conv = L.Convolution(bottom,
# 		num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
# 		# weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0)
# 		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], engine=1)
# 	if phase == 'train':
# 		bn = L.BatchNorm(conv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)], in_place=True)
# 	else:
# 		bn = L.BatchNorm(conv, use_global_stats=1)
# 	scale = L.Scale(bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0), in_place=True)
# 	relu = L.ReLU(scale, in_place=True, engine=1)
# 	return conv, bn, scale, relu
############ ############
# def deconv_relu(bottom, num_output, pad=0, kernel_size=3, stride=1):
# 	deconv = L.Deconvolution(bottom,
# 		param=[dict(lr_mult=1, decay_mult=1)],
# 		convolution_param=dict(num_output=num_output,  pad=pad, kernel_size=kernel_size, stride=stride,
# 			#weight_filler=dict(type='gaussian', std=0.001), bias_term=0,
# 			weight_filler=dict(type='msra'), bias_term=0,
# 			engine=1))
# 	relu = L.ReLU(deconv, in_place=True, engine=1)
# 	return deconv, relu
# ############ ############
# def deconv_bn_scale_relu(bottom, num_output, pad=0, kernel_size=3, stride=1, phase='train'):
# 	deconv = L.Deconvolution(bottom,
# 		param=[dict(lr_mult=1, decay_mult=1)],
# 		convolution_param=dict(num_output=num_output,  pad=pad, kernel_size=kernel_size, stride=stride,
# 			weight_filler=dict(type='gaussian', std=0.001), bias_term=0,
# 			engine=1))
# 	if phase == 'train':
# 		bn = L.BatchNorm(deconv, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)], in_place=True)
# 	else:
# 		bn = L.BatchNorm(deconv, use_global_stats=1)
# 	scale = L.Scale(bn, axis=1, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0), in_place=True)
# 	relu = L.ReLU(scale, in_place=True, engine=1)
# 	return deconv, bn, scale, relu
# ############ ############
# def max_pool(bottom, pad=0, kernel_size=2, stride=2):
#     return L.Pooling(bottom, pool=P.Pooling.MAX, pad=pad, kernel_size=kernel_size, stride=stride)
# ############ ############
# def ave_pool(bottom, pad=0, kernel_size=2, stride=2):
#     return L.Pooling(bottom, pool=P.Pooling.AVE, pad=pad, kernel_size=kernel_size, stride=stride)
# ############ ############
# def max_pool_nd(bottom, pad=[0,0,0], kernel_size=[2,2,2], stride=[2,2,2]):
# 	return L.PoolingND(bottom, pool=0, pad=pad, kernel_size=kernel_size, stride=stride, engine=1)
# ############ ############
# def ave_pool_nd(bottom, pad=[0,0,0], kernel_size=[2,2,2], stride=[2,2,2]):
# 	return L.PoolingND(bottom, pool=1, pad=pad, kernel_size=kernel_size, stride=stride, engine=1)

############ ############ ############ ############ ############ ############

def unet_2d_bn(dim_data, dim_label, num_class, ignore_label=255, phase="train"):
	net = caffe.NetSpec()
	############ d0 ############
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if phase == "train":
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		net.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))	
	net.d0b_conv, net.d0b_bn, net.d0b_scale, net.d0b_relu = conv_bn_scale_relu(net.data, 64, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d0c_conv, net.d0c_bn, net.d0c_scale, net.d0c_relu = conv_bn_scale_relu(net.d0b_relu, 64, pad=1, kernel_size=3, stride=1, phase=phase)
	############ d1 ############
	net.d1a_pool = max_pool(net.d0c_relu, pad=0, kernel_size=2, stride=2)
	net.d1b_conv, net.d1b_bn, net.d1b_scale, net.d1b_relu = conv_bn_scale_relu(net.d1a_pool, 128, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d1c_conv, net.d1c_bn, net.d1c_scale, net.d1c_relu = conv_bn_scale_relu(net.d1b_relu, 128, pad=1, kernel_size=3, stride=1, phase=phase)
	############ d2 ############
	net.d2a_pool = max_pool(net.d1c_relu, pad=0, kernel_size=2, stride=2)
	net.d2b_conv, net.d2b_bn, net.d2b_scale, net.d2b_relu = conv_bn_scale_relu(net.d2a_pool, 256, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d2c_conv, net.d2c_bn, net.d2c_scale, net.d2c_relu = conv_bn_scale_relu(net.d2b_relu, 256, pad=1, kernel_size=3, stride=1, phase=phase)
	############ d3 ############
	net.d3a_pool = max_pool(net.d2c_relu, pad=0, kernel_size=2, stride=2)
	net.d3b_conv, net.d3b_bn, net.d3b_scale, net.d3b_relu = conv_bn_scale_relu(net.d3a_pool, 512, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d3c_conv, net.d3c_bn, net.d3c_scale, net.d3c_relu = conv_bn_scale_relu(net.d3b_relu, 512, pad=1, kernel_size=3, stride=1, phase=phase)
	############ d4 ############
	net.d4a_pool = max_pool(net.d3c_relu, pad=0, kernel_size=2, stride=2)
	net.d4b_conv, net.d4b_bn, net.d4b_scale, net.d4b_relu = conv_bn_scale_relu(net.d4a_pool, 1024, pad=1, kernel_size=3, stride=1, phase=phase)
	net.d4c_conv, net.d4c_bn, net.d4c_scale, net.d4c_relu = conv_bn_scale_relu(net.d4b_relu, 1024, pad=1, kernel_size=3, stride=1, phase=phase)

	############ u3 ############
	### a ### First Deconvolution
	net.u3a_dconv, net.u3a_bn, net.u3a_scale, net.u3a_relu = deconv_bn_scale_relu(net.d4c_relu, 512, pad=0, kernel_size=2, stride=2, phase=phase)
	### b ### Crop and Concat
	net.u3b_crop = L.Crop(net.u3a_relu, net.d3c_relu, axis=2, offset=0)
	#net.u3b_crop = crop(net.u3a_relu, net.d3c_relu)
	net.u3b_concat = L.Concat(net.u3b_crop, net.d3c_relu, axis=1)
	### c ###
	net.u3c_conv, net.u3c_bn, net.u3c_scale, net.u3c_relu = conv_bn_scale_relu(net.u3b_concat, 512, pad=1, kernel_size=3, stride=1, phase=phase)
	### d ###
	net.u3d_conv, net.u3d_bn, net.u3d_scale, net.u3d_relu = conv_bn_scale_relu(net.u3c_relu, 512, pad=1, kernel_size=3, stride=1, phase=phase)

	############ u2 ############
	### a ### Second Deconvolution
	net.u2a_dconv, net.u2a_bn, net.u2a_scale, net.u2a_relu = deconv_bn_scale_relu(net.u3d_relu, 256, pad=0, kernel_size=2, stride=2, phase=phase)
	### b ### Crop and Concat
	net.u2b_crop = L.Crop(net.u2a_relu, net.d2c_relu, axis=2, offset=0)
	#net.u2b_crop = crop(net.u2a_relu, net.d2c_relu)
	net.u2b_concat = L.Concat(net.u2b_crop, net.d2c_relu, axis=1)
	### c ###
	net.u2c_conv, net.u2c_bn, net.u2c_scale, net.u2c_relu = conv_bn_scale_relu(net.u2b_concat, 256, pad=1, kernel_size=3, stride=1, phase=phase)
	### d ###
	net.u2d_conv, net.u2d_bn, net.u2d_scale, net.u2d_relu = conv_bn_scale_relu(net.u2c_relu, 256, pad=1, kernel_size=3, stride=1, phase=phase)

	############ u1 ############
	### a ### Third Deconvolution
	net.u1a_dconv, net.u1a_bn, net.u1a_scale, net.u1a_relu = deconv_bn_scale_relu(net.u2d_relu, 128, pad=0, kernel_size=2, stride=2, phase=phase)
	### b ### Crop and Concat
	net.u1b_crop = L.Crop(net.u1a_relu, net.d1c_relu, axis=2, offset=0)
	#net.u1b_crop = crop(net.u1a_relu, net.d1c_relu)
	net.u1b_concat = L.Concat(net.u1b_crop, net.d1c_relu, axis=1)
	### c ###
	net.u1c_conv, net.u1c_bn, net.u1c_scale, net.u1c_relu = conv_bn_scale_relu(net.u1b_concat, 128, pad=1, kernel_size=3, stride=1, phase=phase)
	### d ###
	net.u1d_conv, net.u1d_bn, net.u1d_scale, net.u1d_relu = conv_bn_scale_relu(net.u1c_relu, 128, pad=1, kernel_size=3, stride=1, phase=phase)

	############ u0 ############
	### a ### Fourth Deconvolution
	net.u0a_dconv, net.u0a_bn, net.u0a_scale, net.u0a_relu = deconv_bn_scale_relu(net.u1d_relu, 64, pad=0, kernel_size=2, stride=2, phase=phase)
	### b ### Crop and Concat
	net.u0b_crop = L.Crop(net.u0a_relu, net.d0c_relu, axis=2, offset=0)
	#net.u0b_crop = crop(net.u0a_relu, net.d0c_relu)
	net.u0b_concat = L.Concat(net.u0b_crop, net.d0c_relu, axis=1)
	### c ###
	net.u0c_conv, net.u0c_bn, net.u0c_scale, net.u0c_relu = conv_bn_scale_relu(net.u0b_concat, 64, pad=1, kernel_size=3, stride=1, phase=phase)
	### d ###
	net.u0d_conv, net.u0d_bn, net.u0d_scale, net.u0d_relu = conv_bn_scale_relu(net.u0c_relu, 64, pad=1, kernel_size=3, stride=1, phase=phase)

	############ Score ############
	net.score = L.Convolution(net.u0d_relu,
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],	
		num_output=num_class, pad=0, kernel_size=1, stride=1, 
		weight_filler=dict(type='gaussian', std=0.001), bias_filler=dict(type='constant', value=0),
		engine=1)

	############ Loss ############
	if phase == "train":
		net.loss = L.WeightedSoftmaxWithLoss(net.score, net.label, net.label_weight,
			phase=0,
			loss_weight=1,
			loss_param=dict(ignore_label=ignore_label))

	return net.to_proto()


def make_resnet(net, dim_data, dim_label, num_class, prototxt_train, prototxt_test):
	"""
	(3, 4,  6, 3) for 50 layers
	(3, 4, 23, 3) for 101 layers
	(3, 8, 26, 3) for 152 layers
	"""
	# register net
	__nets = ['resnet_50', 'resnet_101', 'resnet_152']
	assert net in __nets, 'Unknown net: {}'.format(net)
	global ignore_label
	ignore_label = 255

	if net == 'resnet_50':
		train_net = resnet_50(dim_data, dim_label, num_class, ignore_label, phase="TRAIN")
		test_net = resnet_50(dim_data, dim_label, num_class, ignore_label, phase="TEST")
	if net == 'resnet_101':
		train_net = resnet(dim_data, dim_label, num_class, ignore_label, phase="TRAIN", stages=(3, 4, 23, 3))
		test_net = resnet(dim_data, dim_label, num_class, ignore_label, phase="TEST", stages=(3, 4, 23, 3))
	if net == 'resnet_152':
		train_net = resnet(dim_data, dim_label, num_class, ignore_label, phase="TRAIN", stages=(3, 8, 26, 3))
		test_net = resnet(dim_data, dim_label, num_class, ignore_label, phase="TEST", stages=(3, 8, 26, 3))

	with open(prototxt_train, 'w') as f:
		f.write(str(train_net))

	with open(prototxt_test, 'w') as f:
		f.write(str(test_net))