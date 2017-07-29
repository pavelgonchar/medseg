import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop # crop(net.upscore, net.data) automatically calculate the axis and offset

############ ############
def conv_relu(bottom, num_output, pad=1, kernel_size=3, stride=1,engine=2):
    conv = L.Convolution(bottom,
    	num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
    	# weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0),
    	weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
        engine=engine)
    relu = L.ReLU(conv, in_place=True,engine=engine)
    return conv, relu
############ ############
def conv_bn_scale_relu(bottom, num_output, pad=0, kernel_size=3, stride=1, phase='train',engine=2):
	conv = L.Convolution(bottom,
		num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
		# weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
		engine=engine)
	if phase == 'train':
		bn = L.BatchNorm(conv, in_place=True, use_global_stats=0, param=[dict(lr_mult=0), dict(lr_mult=0), dict(lr_mult=0)])
	else:
		bn = L.BatchNorm(conv, in_place=True, use_global_stats=1)
	scale = L.Scale(bn, axis=1, in_place=True, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	relu = L.ReLU(scale, in_place=True,engine=engine)
	return conv, bn, scale, relu
############ ############
def dila_conv_relu(bottom, num_output, pad=0, kernel_size=3, stride=1, dilation=1,engine=1):
	conv = L.Convolution(bottom,
		num_output=num_output, pad=pad, kernel_size=kernel_size, stride=stride,
		dilation=dilation,
		# weight_filler=dict(type='xavier'), bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)],
		engine=engine)
	relu = L.ReLU(conv, in_place=True,engine=engine)
	return conv, relu
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
	scale = L.Scale(bn, axis=1, in_place=True, filler=dict(type='constant', value=1), bias_term=1, bias_filler=dict(type='constant', value=0))
	relu = L.ReLU(scale, in_place=True,engine=engine)
	return conv, bn, scale, relu
############ ############
def max_pool(bottom, pad=0, kernel_size=2, stride=2,engine=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, pad=pad, kernel_size=kernel_size, stride=stride,engine=engine)
############ ############
def ave_pool(bottom, pad=0, kernel_size=2, stride=2,engine=2):
    return L.Pooling(bottom, pool=P.Pooling.AVE, pad=pad, kernel_size=kernel_size, stride=stride,engine=engine)


############ ############
def deconv_relu(bottom, num_output, pad=0, kernel_size=3, stride=1, engine=2):
	deconv = L.Deconvolution(bottom,
		param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=num_output,  pad=pad, kernel_size=kernel_size, stride=stride,
			#weight_filler=dict(type='gaussian', std=0.001), bias_term=0,
			weight_filler=dict(type='msra'), bias_term=0,
			engine=engine))
	relu = L.ReLU(deconv, in_place=True, engine=engine)
	return deconv, relu

############ ############ ############ ############ ############ ############


def deeplab_v2_vgg_16_unet(dim_data, dim_label, num_class, ignore_label=255, phase="train"):
	net = caffe.NetSpec()
	### Input ###
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if phase=="train":
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		# net.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))

	###################### DeepLab ####################
	### Block 1 ###
	net.conv1_1, net.relu1_1 = conv_relu(net.data, 64, pad=1, kernel_size=3, stride=1)
	net.conv1_2, net.relu1_2 = conv_relu(net.conv1_1, 64, pad=1, kernel_size=3, stride=1)
	net.pool1 = max_pool(net.conv1_2, pad=1, kernel_size=3, stride=2)
	### Block 2 ###
	net.conv2_1, net.relu2_1 = conv_relu(net.pool1, 128, pad=1, kernel_size=3, stride=1)
	net.conv2_2, net.relu2_2 = conv_relu(net.conv2_1, 128, pad=1, kernel_size=3, stride=1)
	net.pool2 = max_pool(net.conv2_2,  pad=1, kernel_size=3, stride=2)
	### Block 3 ###
	net.conv3_1, net.relu3_1 = conv_relu(net.pool2, 256, pad=1, kernel_size=3, stride=1)
	net.conv3_2, net.relu3_2 = conv_relu(net.conv3_1, 256, pad=1, kernel_size=3, stride=1)
	net.conv3_3, net.relu3_3 = conv_relu(net.conv3_2, 256, pad=1, kernel_size=3, stride=1)
	net.pool3 = max_pool(net.conv3_3,  pad=1, kernel_size=3, stride=2)
	### Block 4 ###
	net.conv4_1, net.relu4_1 = conv_relu(net.pool3, 512, pad=1, kernel_size=3, stride=1)
	net.conv4_2, net.relu4_2 = conv_relu(net.conv4_1, 512, pad=1, kernel_size=3, stride=1)
	net.conv4_3, net.relu4_3 = conv_relu(net.conv4_2, 512, pad=1, kernel_size=3, stride=1)
	net.pool4 = max_pool(net.conv4_3, pad=1, kernel_size=3, stride=1)
	### Block 5 ###
	net.conv5_1, net.relu5_1 = dila_conv_relu(net.pool4, 512, pad=2, kernel_size=3, stride=1, dilation=2)
	net.conv5_2, net.relu5_2 = dila_conv_relu(net.conv5_1, 512, pad=2, kernel_size=3, stride=1, dilation=2)
	net.conv5_3, net.relu5_3 = dila_conv_relu(net.conv5_2, 512, pad=2, kernel_size=3, stride=1, dilation=2)
	net.pool5 = max_pool(net.conv5_3, pad=1, kernel_size=3, stride=1)

	###################### DeepLab V2 ####################
	### Additional Average Pool ###
	# net.pool5a = ave_pool(net.pool5, pad=1, kernel_size=3, stride=1)

	### Block 6 ###
	### hole = 6, branch 1
	net.fc6_1, net.relu6_1 = dila_conv_relu(net.pool5, 1024, pad=6, kernel_size=3, stride=1, dilation=6)
	net.drop6_1 = L.Dropout(net.fc6_1, dropout_ratio=0.5, in_place=True)
	### hole = 12, branch 2
	net.fc6_2, net.relu6_2 = dila_conv_relu(net.pool5, 1024, pad=12, kernel_size=3, stride=1, dilation=12)
	net.drop6_2 = L.Dropout(net.fc6_2, dropout_ratio=0.5, in_place=True)
	### hole = 18, branch 3
	net.fc6_3, net.relu6_3 = dila_conv_relu(net.pool5, 1024, pad=18, kernel_size=3, stride=1, dilation=18)
	net.drop6_3 = L.Dropout(net.fc6_3, dropout_ratio=0.5, in_place=True)
	### hole = 24, branch 4
	net.fc6_4, net.relu6_4 = dila_conv_relu(net.pool5, 1024, pad=24, kernel_size=3, stride=1, dilation=24)
	net.drop6_4 = L.Dropout(net.fc6_4, dropout_ratio=0.5, in_place=True)

	### Block 7 ###
	net.fc7_1, net.relu7_1 = conv_relu(net.fc6_1, 1024, pad=0, kernel_size=1, stride=1)
	net.drop7_1 = L.Dropout(net.fc7_1, dropout_ratio=0.5, in_place=True)
	net.fc7_2, net.relu7_2 = conv_relu(net.fc6_2, 1024, pad=0, kernel_size=1, stride=1)
	net.drop7_2 = L.Dropout(net.fc7_2, dropout_ratio=0.5, in_place=True)
	net.fc7_3, net.relu7_3 = conv_relu(net.fc6_3, 1024, pad=0, kernel_size=1, stride=1)
	net.drop7_3 = L.Dropout(net.fc7_3, dropout_ratio=0.5, in_place=True)
	net.fc7_4, net.relu7_4 = conv_relu(net.fc6_4, 1024, pad=0, kernel_size=1, stride=1)
	net.drop7_4 = L.Dropout(net.fc7_4, dropout_ratio=0.5, in_place=True)

	### Block 8 ###
	net.fc8_1 = L.Convolution(net.fc7_1,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],
		num_output=num_class,
		pad=0, kernel_size=1, stride=1,
		# weight_filler=dict(type='gaussian', std=0.01),bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=2)
	net.fc8_2 = L.Convolution(net.fc7_2,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],
		num_output=num_class,
		pad=0, kernel_size=1, stride=1,
		# weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=2)
	net.fc8_3 = L.Convolution(net.fc7_3,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],
		num_output=num_class,
		pad=0, kernel_size=1, stride=1,
		# weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=2)
	net.fc8_4 = L.Convolution(net.fc7_4,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],
		num_output=num_class,
		pad=0, kernel_size=1, stride=1,
		# weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=2)
	### SUM the four branches ###
	# PROD = 0, # SUM = 1, # MAX = 2;
	net.fc8 = L.Eltwise(net.fc8_1, net.fc8_2, net.fc8_3, net.fc8_4,
		eltwise_param=dict(operation=1))
	net.fc8_relu = L.ReLU(net.fc8, in_place=True, engine=2)

	###
	### a ### First Deconvolution
	############ u2 ############
	net.u2a_dconv, net.u2a_relu = deconv_relu(net.fc8, 256, pad=0, kernel_size=2, stride=2)
	### b ### Crop and Concat
	net.u2b_crop = L.Crop(net.u2a_dconv, net.conv3_3, axis=2, offset=0)
	net.u2b_concat = L.Concat(net.u2b_crop, net.conv3_3, axis=1)
	### c ###
	net.u2c_conv, net.u2c_relu = conv_relu(net.u2b_concat, 256, pad=1, kernel_size=3, stride=1)
	### d ###
	net.u2d_conv, net.u2d_relu = conv_relu(net.u2c_conv, 256, pad=1, kernel_size=3, stride=1)

	############ u1 ############
	### a ### Second Deconvolution
	net.u1a_dconv, net.u1a_relu = deconv_relu(net.u2d_conv, 128, pad=0, kernel_size=2, stride=2)
	### b ### Crop and Concat
	net.u1b_crop = L.Crop(net.u1a_dconv, net.conv2_2, axis=2, offset=0)
	net.u1b_concat = L.Concat(net.u1b_crop, net.conv2_2, axis=1)
	### c ###
	net.u1c_conv, net.u1c_relu = conv_relu(net.u1b_concat, 128, pad=1, kernel_size=3, stride=1)
	### d ###
	net.u1d_conv, net.u1d_relu = conv_relu(net.u1c_conv, 128, pad=1, kernel_size=3, stride=1)

	############ u0 ############
	### a ### Third Deconvolution
	net.u0a_dconv, net.u0a_relu = deconv_relu(net.u1d_conv, 64, pad=0, kernel_size=2, stride=2)
	### b ### Crop and Concat
	net.u0b_crop = L.Crop(net.u0a_dconv, net.conv1_2, axis=2, offset=1)
	net.u0b_concat = L.Concat(net.u0b_crop, net.conv1_2, axis=1)
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
		engine=2)

	# ### Block 9 Deconvolution ###
	# net.upscore = L.Deconvolution(net.fc8,
	# 	param=[dict(lr_mult=1, decay_mult=1)],
	# 	convolution_param=dict(num_output=num_class,
	# 		pad=4, kernel_size=16, stride=8,
	# 		weight_filler=dict(type='bilinear'),
	# 		bias_term=0, engine=2))

	### Crop ###
	# net.score = L.Crop(net.upscore, net.data, axis=2, offset=3)
	# net.score = crop(net.upscore, net.data)

	### Loss ###
	if phase=="train":
		net.loss = L.SoftmaxWithLoss(net.score, net.label,
			phase=0,
			loss_weight=1,
			loss_param=dict(ignore_label=ignore_label))
		# net.loss = L.WeightedSoftmaxWithLoss(net.score, net.label, net.label_weight,
		# 	phase=0,
		# 	loss_weight=1,
		# 	loss_param=dict(ignore_label=ignore_label))

	return net.to_proto()


def deeplab_largefov(dim_data, dim_label, num_class, ignore_label=255, phase="train"):
	net = caffe.NetSpec()
	### Input ###
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if phase=="train":
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		# net.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))

	###################### DeepLab ####################
	### Block 1 ###
	net.conv1_1, net.relu1_1 = conv_relu(net.data, 64, pad=1, kernel_size=3, stride=1)
	net.conv1_2, net.relu1_2 = conv_relu(net.conv1_1, 64, pad=1, kernel_size=3, stride=1)
	net.pool1 = max_pool(net.conv1_2, pad=1, kernel_size=3, stride=2)
	### Block 2 ###
	net.conv2_1, net.relu2_1 = conv_relu(net.pool1, 128, pad=1, kernel_size=3, stride=1)
	net.conv2_2, net.relu2_2 = conv_relu(net.conv2_1, 128, pad=1, kernel_size=3, stride=1)
	net.pool2 = max_pool(net.conv2_2,  pad=1, kernel_size=3, stride=2)
	### Block 3 ###
	net.conv3_1, net.relu3_1 = conv_relu(net.pool2, 256, pad=1, kernel_size=3, stride=1)
	net.conv3_2, net.relu3_2 = conv_relu(net.conv3_1, 256, pad=1, kernel_size=3, stride=1)
	net.conv3_3, net.relu3_3 = conv_relu(net.conv3_2, 256, pad=1, kernel_size=3, stride=1)
	net.pool3 = max_pool(net.conv3_3,  pad=1, kernel_size=3, stride=2)
	### Block 4 ###
	net.conv4_1, net.relu4_1 = conv_relu(net.pool3, 512, pad=1, kernel_size=3, stride=1)
	net.conv4_2, net.relu4_2 = conv_relu(net.conv4_1, 512, pad=1, kernel_size=3, stride=1)
	net.conv4_3, net.relu4_3 = conv_relu(net.conv4_2, 512, pad=1, kernel_size=3, stride=1)
	net.pool4 = max_pool(net.conv4_3, pad=1, kernel_size=3, stride=1)
	### Block 5 ###
	net.conv5_1, net.relu5_1 = dila_conv_relu(net.pool4, 512, pad=2, kernel_size=3, stride=1, dilation=2)
	net.conv5_2, net.relu5_2 = dila_conv_relu(net.conv5_1, 512, pad=2, kernel_size=3, stride=1, dilation=2)
	net.conv5_3, net.relu5_3 = dila_conv_relu(net.conv5_2, 512, pad=2, kernel_size=3, stride=1, dilation=2)
	net.pool5 = max_pool(net.conv5_3, pad=1, kernel_size=3, stride=1)

	### Additional Average Pool ###
	net.pool5a = ave_pool(net.pool5, pad=1, kernel_size=3, stride=1)

	### Block 6 ###
	net.fc6, net.relu6 = dila_conv_relu(net.pool5a, 1024, pad=12, kernel_size=3, stride=1, dilation=12)
	net.drop6 = L.Dropout(net.fc6, dropout_ratio=0.5, in_place=True)

	### Block 7 ###
	net.fc7, net.relu7 = conv_relu(net.fc6, 1024, pad=0, kernel_size=1, stride=1)
	net.drop7 = L.Dropout(net.fc7, dropout_ratio=0.5, in_place=True)

	### Block 8 ###
	net.score_fr = L.Convolution(net.fc7,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],
		num_output=num_class,
		pad=0, kernel_size=1, stride=1,
		# weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=2)

	### Block 9 Deconvolution Upsample the score_fr ###
	net.upscore = L.Deconvolution(net.score_fr,
		param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=num_class,
			pad=4, kernel_size=16, stride=8,
			weight_filler=dict(type='bilinear'),bias_term=0,
			engine=2))

	### Crop ###
	net.score = L.Crop(net.upscore, net.data, axis=2, offset=3)
	# net.score = crop(net.upscore, net.data)

	### Loss ###
	if phase=="train":
		net.loss = L.SoftmaxWithLoss(net.score, net.label,
			phase=0,
			loss_weight=1,
			loss_param=dict(ignore_label=ignore_label))
		# net.loss = L.WeightedSoftmaxWithLoss(net.score, net.label, net.label_weight,
		# 	phase=0,
		# 	loss_weight=1,
		# 	loss_param=dict(ignore_label=ignore_label))

	return net.to_proto()

def deeplab_largefov_bn(im_data, dim_label, num_class, ignore_label=255, phase="train"):
	net = caffe.NetSpec()
	###################### Input ####################
	net.data = L.Input(input_param=dict(shape=dict(dim=im_data)))
	if phase=="train":
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		# net.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))

	###################### DeepLab ####################
	### Block 1 ###
	net.conv1_1, net.bn1_1, net.scale1_1, net.relu1_1 = conv_bn_scale_relu(net.data, 64, pad=1, kernel_size=3, stride=1, phase=phase)
	net.conv1_2, net.bn1_2, net.scale1_2, net.relu1_2 = conv_bn_scale_relu(net.conv1_1, 64, pad=1, kernel_size=3, stride=1, phase=phase)
	net.pool1 = max_pool(net.conv1_2, pad=1, kernel_size=3, stride=2)
	### Block 2 ###
	net.conv2_1, net.bn2_1, net.scale2_1, net.relu2_1 = conv_bn_scale_relu(net.pool1, 128, pad=1, kernel_size=3, stride=1, phase=phase)
	net.conv2_2, net.bn2_2, net.scale2_2, net.relu2_2 = conv_bn_scale_relu(net.conv2_1, 128, pad=1, kernel_size=3, stride=1, phase=phase)
	net.pool2 = max_pool(net.conv2_2, pad=1, kernel_size=3, stride=2)
	### Block 3 ###
	net.conv3_1, net.bn3_1, net.scale3_1, net.relu3_1 = conv_bn_scale_relu(net.pool2, 256, pad=1, kernel_size=3, stride=1, phase=phase)
	net.conv3_2, net.bn3_2, net.scale3_2, net.relu3_2 = conv_bn_scale_relu(net.conv3_1, 256, pad=1, kernel_size=3, stride=1, phase=phase)
	net.conv3_3, net.bn3_3, net.scale3_3, net.relu3_3 = conv_bn_scale_relu(net.conv3_2, 256, pad=1, kernel_size=3, stride=1, phase=phase)
	net.pool3 = max_pool(net.conv3_3, pad=1, kernel_size=3, stride=2)
	### Block 4 ###
	net.conv4_1, net.bn4_1, net.scale4_1, net.relu4_1 = conv_bn_scale_relu(net.pool3, 512, pad=1, kernel_size=3, stride=1, phase=phase)
	net.conv4_2, net.bn4_2, net.scale4_2, net.relu4_2 = conv_bn_scale_relu(net.conv4_1, 512, pad=1, kernel_size=3, stride=1, phase=phase)
	net.conv4_3, net.bn4_3, net.scale4_3, net.relu4_3 = conv_bn_scale_relu(net.conv4_2, 512, pad=1, kernel_size=3, stride=1, phase=phase)
	net.pool4 = max_pool(net.conv4_3, pad=1, kernel_size=3, stride=1)
	### Block 5 ###
	net.conv5_1, net.bn5_1, net.scale5_1, net.relu5_1 = dila_conv_bn_scale_relu(net.pool4, 512, pad=2, kernel_size=3, stride=1, dilation=2, phase=phase)
	net.conv5_2, net.bn5_2, net.scale5_2, net.relu5_2 = dila_conv_bn_scale_relu(net.conv5_1, 512, pad=2, kernel_size=3, stride=1, dilation=2, phase=phase)
	net.conv5_3, net.bn5_3, net.scale5_3, net.relu5_3 = dila_conv_bn_scale_relu(net.conv5_2, 512, pad=2, kernel_size=3, stride=1, dilation=2, phase=phase)
	net.pool5 = max_pool(net.conv5_3, pad=1, kernel_size=3, stride=1)

	### Additional Average Pool ###
	net.pool5a = ave_pool(net.pool5, pad=1, kernel_size=3, stride=1)

	### Block 6 ###
	net.fc6, net.bn6, net.scale6, net.relu6 = dila_conv_bn_scale_relu(net.pool5a, 1024, pad=12, kernel_size=3, stride=1, dilation=12, phase=phase)
	net.drop6 = L.Dropout(net.fc6, dropout_ratio=0.5, in_place=True)

	### Block 7 ###
	net.fc7, net.bn7, net.scale7, net.relu7 = conv_bn_scale_relu(net.fc6, 1024, pad=0, kernel_size=1, stride=1, phase=phase)
	net.drop7 = L.Dropout(net.fc7, dropout_ratio=0.5, in_place=True)

	### Block 8 ###
	net.score_fr = L.Convolution(net.fc7,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],
		num_output=num_class, pad=0, kernel_size=1, stride=1,
		# weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=2)

	### Block 9 Deconvolution Upsample the score_fr ###
	net.upscore = L.Deconvolution(net.score_fr,
		param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=num_class,
			pad=4, kernel_size=16, stride=8,
			weight_filler=dict(type='bilinear'), bias_term=0,
			engine=2))

	### Crop ###
	net.score = L.Crop(net.upscore, net.data, axis=2, offset=3)
	# net.score = crop(net.upscore, net.data)

	### Loss ###
	if phase=="train":
		net.loss = L.SoftmaxWithLoss(net.score, net.label,
			phase=0,
			loss_weight=1,
			loss_param=dict(ignore_label=ignore_label))
		# net.loss = L.WeightedSoftmaxWithLoss(net.score, net.label, net.label_weight,
		# 	phase=0,
		# 	loss_weight=1,
		# 	loss_param=dict(ignore_label=ignore_label))

	return net.to_proto()

def deeplab_v2_vgg_16(dim_data, dim_label, num_class, ignore_label=255, phase="train"):
	net = caffe.NetSpec()
	### Input ###
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if phase=="train":
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		# net.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))

	###################### DeepLab ####################
	### Block 1 ###
	net.conv1_1, net.relu1_1 = conv_relu(net.data, 64, pad=1, kernel_size=3, stride=1)
	net.conv1_2, net.relu1_2 = conv_relu(net.conv1_1, 64, pad=1, kernel_size=3, stride=1)
	net.pool1 = max_pool(net.conv1_2, pad=1, kernel_size=3, stride=2)
	### Block 2 ###
	net.conv2_1, net.relu2_1 = conv_relu(net.pool1, 128, pad=1, kernel_size=3, stride=1)
	net.conv2_2, net.relu2_2 = conv_relu(net.conv2_1, 128, pad=1, kernel_size=3, stride=1)
	net.pool2 = max_pool(net.conv2_2,  pad=1, kernel_size=3, stride=2)
	### Block 3 ###
	net.conv3_1, net.relu3_1 = conv_relu(net.pool2, 256, pad=1, kernel_size=3, stride=1)
	net.conv3_2, net.relu3_2 = conv_relu(net.conv3_1, 256, pad=1, kernel_size=3, stride=1)
	net.conv3_3, net.relu3_3 = conv_relu(net.conv3_2, 256, pad=1, kernel_size=3, stride=1)
	net.pool3 = max_pool(net.conv3_3,  pad=1, kernel_size=3, stride=2)
	### Block 4 ###
	net.conv4_1, net.relu4_1 = conv_relu(net.pool3, 512, pad=1, kernel_size=3, stride=1)
	net.conv4_2, net.relu4_2 = conv_relu(net.conv4_1, 512, pad=1, kernel_size=3, stride=1)
	net.conv4_3, net.relu4_3 = conv_relu(net.conv4_2, 512, pad=1, kernel_size=3, stride=1)
	net.pool4 = max_pool(net.conv4_3, pad=1, kernel_size=3, stride=1)
	### Block 5 ###
	net.conv5_1, net.relu5_1 = dila_conv_relu(net.pool4, 512, pad=2, kernel_size=3, stride=1, dilation=2)
	net.conv5_2, net.relu5_2 = dila_conv_relu(net.conv5_1, 512, pad=2, kernel_size=3, stride=1, dilation=2)
	net.conv5_3, net.relu5_3 = dila_conv_relu(net.conv5_2, 512, pad=2, kernel_size=3, stride=1, dilation=2)
	net.pool5 = max_pool(net.conv5_3, pad=1, kernel_size=3, stride=1)

	###################### DeepLab V2 ####################
	### Additional Average Pool ###
	# net.pool5a = ave_pool(net.pool5, pad=1, kernel_size=3, stride=1)

	### Block 6 ###
	### hole = 6, branch 1
	net.fc6_1, net.relu6_1 = dila_conv_relu(net.pool5, 1024, pad=6, kernel_size=3, stride=1, dilation=6)
	net.drop6_1 = L.Dropout(net.fc6_1, dropout_ratio=0.5, in_place=True)
	### hole = 12, branch 2
	net.fc6_2, net.relu6_2 = dila_conv_relu(net.pool5, 1024, pad=12, kernel_size=3, stride=1, dilation=12)
	net.drop6_2 = L.Dropout(net.fc6_2, dropout_ratio=0.5, in_place=True)
	### hole = 18, branch 3
	net.fc6_3, net.relu6_3 = dila_conv_relu(net.pool5, 1024, pad=18, kernel_size=3, stride=1, dilation=18)
	net.drop6_3 = L.Dropout(net.fc6_3, dropout_ratio=0.5, in_place=True)
	### hole = 24, branch 4
	net.fc6_4, net.relu6_4 = dila_conv_relu(net.pool5, 1024, pad=24, kernel_size=3, stride=1, dilation=24)
	net.drop6_4 = L.Dropout(net.fc6_4, dropout_ratio=0.5, in_place=True)

	### Block 7 ###
	net.fc7_1, net.relu7_1 = conv_relu(net.fc6_1, 1024, pad=0, kernel_size=1, stride=1)
	net.drop7_1 = L.Dropout(net.fc7_1, dropout_ratio=0.5, in_place=True)
	net.fc7_2, net.relu7_2 = conv_relu(net.fc6_2, 1024, pad=0, kernel_size=1, stride=1)
	net.drop7_2 = L.Dropout(net.fc7_2, dropout_ratio=0.5, in_place=True)
	net.fc7_3, net.relu7_3 = conv_relu(net.fc6_3, 1024, pad=0, kernel_size=1, stride=1)
	net.drop7_3 = L.Dropout(net.fc7_3, dropout_ratio=0.5, in_place=True)
	net.fc7_4, net.relu7_4 = conv_relu(net.fc6_4, 1024, pad=0, kernel_size=1, stride=1)
	net.drop7_4 = L.Dropout(net.fc7_4, dropout_ratio=0.5, in_place=True)

	### Block 8 ###
	net.fc8_1 = L.Convolution(net.fc7_1,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],
		num_output=num_class,
		pad=0, kernel_size=1, stride=1,
		# weight_filler=dict(type='gaussian', std=0.01),bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=2)
	net.fc8_2 = L.Convolution(net.fc7_2,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],
		num_output=num_class,
		pad=0, kernel_size=1, stride=1,
		# weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=2)
	net.fc8_3 = L.Convolution(net.fc7_3,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],
		num_output=num_class,
		pad=0, kernel_size=1, stride=1,
		# weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=2)
	net.fc8_4 = L.Convolution(net.fc7_4,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],
		num_output=num_class,
		pad=0, kernel_size=1, stride=1,
		# weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=2)
	### SUM the four branches ###
	# PROD = 0, # SUM = 1, # MAX = 2;
	net.fc8 = L.Eltwise(net.fc8_1, net.fc8_2, net.fc8_3, net.fc8_4,
		eltwise_param=dict(operation=1))

	### Block 9 Deconvolution ###
	net.upscore = L.Deconvolution(net.fc8,
		param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=num_class,
			pad=4, kernel_size=16, stride=8,
			weight_filler=dict(type='bilinear'),
			bias_term=0, engine=2))

	### Crop ###
	net.score = L.Crop(net.upscore, net.data, axis=2, offset=3)
	# net.score = crop(net.upscore, net.data)

	### Loss ###
	if phase=="train":
		net.loss = L.SoftmaxWithLoss(net.score, net.label,
			phase=0,
			loss_weight=1,
			loss_param=dict(ignore_label=ignore_label))
		# net.loss = L.WeightedSoftmaxWithLoss(net.score, net.label, net.label_weight,
		# 	phase=0,
		# 	loss_weight=1,
		# 	loss_param=dict(ignore_label=ignore_label))

	return net.to_proto()

def deeplab_v2_vgg_16_bn(dim_data, dim_label, num_class, ignore_label=255, phase="train"):
	net = caffe.NetSpec()
	### Input ###
	net.data = L.Input(input_param=dict(shape=dict(dim=dim_data)))
	if phase=="train":
		net.label = L.Input(input_param=dict(shape=dict(dim=dim_label)))
		# net.label_weight = L.Input(input_param=dict(shape=dict(dim=dim_label)))

	###################### DeepLab ####################
	### Block 1 ###
	net.conv1_1, net.bn1_1, net.scale1_1, net.relu1_1 = conv_bn_scale_relu(net.data, 64, pad=1, kernel_size=3, stride=1, phase=phase)
	net.conv1_2, net.bn1_2, net.scale1_2, net.relu1_2 = conv_bn_scale_relu(net.conv1_1, 64, pad=1, kernel_size=3, stride=1, phase=phase)
	net.pool1 = max_pool(net.conv1_2, pad=1, kernel_size=3, stride=2)
	### Block 2 ###
	net.conv2_1, net.bn2_1, net.scale2_1, net.relu2_1 = conv_bn_scale_relu(net.pool1, 128, pad=1, kernel_size=3, stride=1, phase=phase)
	net.conv2_2, net.bn2_2, net.scale2_2, net.relu2_2 = conv_bn_scale_relu(net.conv2_1, 128, pad=1, kernel_size=3, stride=1, phase=phase)
	net.pool2 = max_pool(net.conv2_2,  pad=1, kernel_size=3, stride=2)
	### Block 3 ###
	net.conv3_1, net.bn3_1, net.scale3_1, net.relu3_1 = conv_bn_scale_relu(net.pool2, 256, pad=1, kernel_size=3, stride=1, phase=phase)
	net.conv3_2, net.bn3_2, net.scale3_2, net.relu3_2 = conv_bn_scale_relu(net.conv3_1, 256, pad=1, kernel_size=3, stride=1, phase=phase)
	net.conv3_3, net.bn3_3, net.scale3_3, net.relu3_3 = conv_bn_scale_relu(net.conv3_2, 256, pad=1, kernel_size=3, stride=1, phase=phase)
	net.pool3 = max_pool(net.conv3_3,  pad=1, kernel_size=3, stride=2)
	### Block 4 ###
	net.conv4_1, net.bn4_1, net.scale4_1, net.relu4_1 = conv_bn_scale_relu(net.pool3, 512, pad=1, kernel_size=3, stride=1, phase=phase)
	net.conv4_2, net.bn4_2, net.scale4_2, net.relu4_2 = conv_bn_scale_relu(net.conv4_1, 512, pad=1, kernel_size=3, stride=1, phase=phase)
	net.conv4_3, net.bn4_3, net.scale4_3, net.relu4_3 = conv_bn_scale_relu(net.conv4_2, 512, pad=1, kernel_size=3, stride=1, phase=phase)
	net.pool4 = max_pool(net.conv4_3, pad=1, kernel_size=3, stride=1)
	### Block 5 ###
	net.conv5_1, net.bn5_1, net.scale5_1, net.relu5_1 = dila_conv_bn_scale_relu(net.pool4, 512, pad=2, kernel_size=3, stride=1, dilation=2, phase=phase)
	net.conv5_2, net.bn5_2, net.scale5_2, net.relu5_2 = dila_conv_bn_scale_relu(net.conv5_1, 512, pad=2, kernel_size=3, stride=1, dilation=2, phase=phase)
	net.conv5_3, net.bn5_3, net.scale5_3, net.relu5_3 = dila_conv_bn_scale_relu(net.conv5_2, 512, pad=2, kernel_size=3, stride=1, dilation=2, phase=phase)
	net.pool5 = max_pool(net.conv5_3, pad=1, kernel_size=3, stride=1)

	###################### DeepLab V2 ####################
	### Additional Average Pool ###
	# net.pool5a = ave_pool(net.pool5, pad=1, kernel_size=3, stride=1)

	### Block 6 ###
	### hole = 6, branch 1
	net.fc6_1, net.bn6_1, net.scale6_1, net.relu6_1 = dila_conv_bn_scale_relu(net.pool5, 1024, pad=6, kernel_size=3, stride=1, dilation=6, phase=phase)
	net.drop6_1 = L.Dropout(net.fc6_1, dropout_ratio=0.5, in_place=True)
	### hole = 12, branch 2
	net.fc6_2, net.bn6_2, net.scale6_2, net.relu6_2 = dila_conv_bn_scale_relu(net.pool5, 1024, pad=12, kernel_size=3, stride=1, dilation=12, phase=phase)
	net.drop6_2 = L.Dropout(net.fc6_2, dropout_ratio=0.5, in_place=True)
	### hole = 18, branch 3
	net.fc6_3, net.bn6_3, net.scale6_3, net.relu6_3 = dila_conv_bn_scale_relu(net.pool5, 1024, pad=18, kernel_size=3, stride=1, dilation=18, phase=phase)
	net.drop6_3 = L.Dropout(net.fc6_3, dropout_ratio=0.5, in_place=True)
	### hole = 24, branch 4
	net.fc6_4, net.bn6_4, net.scale6_4, net.relu6_4 = dila_conv_bn_scale_relu(net.pool5, 1024, pad=24, kernel_size=3, stride=1, dilation=24, phase=phase)
	net.drop6_4 = L.Dropout(net.fc6_4, dropout_ratio=0.5, in_place=True)

	### Block 7 ###
	net.fc7_1, net.bn7_1, net.scale7_1, net.relu7_1 = conv_bn_scale_relu(net.fc6_1, 1024, pad=0, kernel_size=1, stride=1, phase=phase)
	net.drop7_1 = L.Dropout(net.fc7_1, dropout_ratio=0.5, in_place=True)
	net.fc7_2, net.bn7_2, net.scale7_2, net.relu7_2 = conv_bn_scale_relu(net.fc6_2, 1024, pad=0, kernel_size=1, stride=1, phase=phase)
	net.drop7_2 = L.Dropout(net.fc7_2, dropout_ratio=0.5, in_place=True)
	net.fc7_3, net.bn7_3, net.scale7_3, net.relu7_3 = conv_bn_scale_relu(net.fc6_3, 1024, pad=0, kernel_size=1, stride=1, phase=phase)
	net.drop7_3 = L.Dropout(net.fc7_3, dropout_ratio=0.5, in_place=True)
	net.fc7_4, net.bn7_4, net.scale7_4, net.relu7_4 = conv_bn_scale_relu(net.fc6_4, 1024, pad=0, kernel_size=1, stride=1, phase=phase)
	net.drop7_4 = L.Dropout(net.fc7_4, dropout_ratio=0.5, in_place=True)

	### Block 8 ###
	net.fc8_1 = L.Convolution(net.fc7_1,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],
		num_output=num_class,
		pad=0, kernel_size=1, stride=1,
		# weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=2)
	net.fc8_2 = L.Convolution(net.fc7_2,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],
		num_output=num_class,
		pad=0, kernel_size=1, stride=1,
		# weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=2)
	net.fc8_3 = L.Convolution(net.fc7_3,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],
		num_output=num_class,
		pad=0, kernel_size=1, stride=1,
		# weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=2)
	net.fc8_4 = L.Convolution(net.fc7_4,
		param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)],
		num_output=num_class,
		pad=0, kernel_size=1, stride=1,
		# weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0),
		weight_filler=dict(type='msra'), bias_filler=dict(type='constant', value=0),
		engine=2)
	### SUM the four branches ###
	# PROD = 0, # SUM = 1, # MAX = 2;
	net.fc8 = L.Eltwise(net.fc8_1, net.fc8_2, net.fc8_3, net.fc8_4,
		eltwise_param=dict(operation=1))

	### Block 9 Deconvolution ###
	net.upscore = L.Deconvolution(net.fc8,
		param=[dict(lr_mult=1, decay_mult=1)],
		convolution_param=dict(num_output=num_class,
			pad=4, kernel_size=16, stride=8,
			weight_filler=dict(type='bilinear'),
			bias_term=0,engine=2))

	### Crop ###
	net.score = L.Crop(net.upscore, net.data, axis=2, offset=4)
	# net.score = crop(net.upscore, net.data)

	### Loss ###
	if phase=="train":
		net.loss = L.SoftmaxWithLoss(net.score, net.label,
			phase=0,
			loss_weight=1,
			loss_param=dict(ignore_label=ignore_label))
		# net.loss = L.WeightedSoftmaxWithLoss(net.score, net.label, net.label_weight,
		# 	phase=0,
		# 	loss_weight=1,
		# 	loss_param=dict(ignore_label=ignore_label))

	return net.to_proto()


def make_deeplab(net, dim_data, dim_label, num_class, prototxt_train, prototxt_test):
	# register net
	__nets = ['deeplab_largefov', 'deeplab_largefov_bn', 'deeplab_v2_vgg_16', 'deeplab_v2_vgg_16_bn', 'deeplab_v2_vgg_16_unet']
	assert net in __nets, 'Unknown net: {}'.format(net)
	global ignore_label
	ignore_label = 255

	if net == 'deeplab_largefov':
		train_net = deeplab_largefov(dim_data, dim_label, num_class, ignore_label, phase="train")
		test_net = deeplab_largefov(dim_data, dim_label, num_class, ignore_label, phase="test")
	if net == 'deeplab_largefov_bn':
		train_net = deeplab_largefov_bn(dim_data, dim_label, num_class, ignore_label, phase="train")
		test_net = deeplab_largefov_bn(dim_data, dim_label, num_class, ignore_label, phase="test")
	if net == 'deeplab_v2_vgg_16':
		train_net = deeplab_v2_vgg_16(dim_data, dim_label, num_class, ignore_label, phase="train")
		test_net = deeplab_v2_vgg_16(dim_data, dim_label, num_class, ignore_label, phase="test")
	if net == 'deeplab_v2_vgg_16_bn':
		train_net = deeplab_v2_vgg_16_bn(dim_data, dim_label, num_class, ignore_label, phase="train")
		test_net = deeplab_v2_vgg_16_bn(dim_data, dim_label, num_class, ignore_label, phase="test")

	if net == 'deeplab_v2_vgg_16_unet':
		train_net = deeplab_v2_vgg_16_unet(dim_data, dim_label, num_class, ignore_label, phase="train")
		test_net = deeplab_v2_vgg_16_unet(dim_data, dim_label, num_class, ignore_label, phase="test")


	with open(prototxt_train, 'w') as f:
		f.write(str(train_net))

	with open(prototxt_test, 'w') as f:
		f.write(str(test_net))
