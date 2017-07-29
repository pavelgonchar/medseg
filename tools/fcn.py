import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop
############ ############
def conv_relu(bottom, num_output, pad=1, kernel_size=3, stride=1):
    conv = L.Convolution(bottom, 
        num_output=num_output,
        pad=pad, kernel_size=kernel_size, stride=stride,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    return conv, L.ReLU(conv, in_place=True)
############ ############
def max_pool(bottom, pad=0, kernel_size=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, pad=pad, kernel_size=kernel_size, stride=stride)
############ ############
def ave_pool(bottom, pad=0, kernel_size=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.AVE, pad=pad, kernel_size=kernel_size, stride=stride)

def fcn_32s(input_dims, num_class, ignore_label=255, phase="train"):
    n = caffe.NetSpec()
    # pydata_params = dict(split=split, mean=(104.00699, 116.66877, 122.67892),
    #         seed=1337)
    # if split == 'train':
    #     pydata_params['sbdd_dir'] = '../data/sbdd/dataset'
    #     pylayer = 'SBDDSegDataLayer'
    # else:
    #     pydata_params['voc_dir'] = '../data/pascal/VOC2011'
    #     pylayer = 'VOCSegDataLayer'
    # n.data, n.label = L.Python(module='voc_layers', layer=pylayer,
    #         ntop=2, param_str=str(pydata_params))
    n.data = L.Input(input_param=dict(shape=dict(dim=input_dims)))
    if phase == 'train':
        n.label = L.Input(phase=0,input_param=dict(shape=dict(dim=input_dims)))
        #n.label_weight = L.Input(input_param=dict(shape=dict(dim=input_dims)))

    # the base net
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64, pad=100, kernel_size=3, stride=1)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64, pad=1, kernel_size=3, stride=1)
    n.pool1 = max_pool(n.relu1_2, pad=0, kernel_size=2, stride=2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128, pad=1, kernel_size=3, stride=1)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128, pad=1, kernel_size=3, stride=1)
    n.pool2 = max_pool(n.relu2_2, pad=0, kernel_size=2, stride=2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256, pad=1, kernel_size=3, stride=1)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256, pad=1, kernel_size=3, stride=1)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256, pad=1, kernel_size=3, stride=1)
    n.pool3 = max_pool(n.relu3_3, pad=0, kernel_size=2, stride=2)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512, pad=1, kernel_size=3, stride=1)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512, pad=1, kernel_size=3, stride=1)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512, pad=1, kernel_size=3, stride=1)
    n.pool4 = max_pool(n.relu4_3, pad=0, kernel_size=2, stride=2)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512, pad=1, kernel_size=3, stride=1)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512, pad=1, kernel_size=3, stride=1)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512, pad=1, kernel_size=3, stride=1)
    n.pool5 = max_pool(n.relu5_3, pad=0, kernel_size=2, stride=2)

    # fully conv
    n.fc6, n.relu6 = conv_relu(n.pool5, 4096, pad=0, kernel_size=7, stride=1)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
    n.fc7, n.relu7 = conv_relu(n.drop6, 4096, pad=0, kernel_size=1, stride=1)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)
    # score
    n.score_fr = L.Convolution(n.drop7, num_output=num_class, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.upscore = L.Deconvolution(n.score_fr,
        convolution_param=dict(num_output=num_class, kernel_size=64, stride=32,
            bias_term=False),
        param=[dict(lr_mult=0)])
    n.score = crop(n.upscore, n.data)
    if phase == 'train':
        n.loss = L.SoftmaxWithLoss(n.score, n.label,
            loss_param=dict(normalize=False, ignore_label=ignore_label))
        #n.loss = L.SoftmaxWithLoss(n.score, n.label, loss_param=dict(ignore_label=ignore_label))

    return n.to_proto()

def fcn_16s(input_dims, num_class, ignore_label=255, phase="train"):
    n = caffe.NetSpec()
    # pydata_params = dict(split=split, mean=(104.00699, 116.66877, 122.67892),
    #         seed=1337)
    # if split == 'train':
    #     pydata_params['sbdd_dir'] = '../../data/sbdd/dataset'
    #     pylayer = 'SBDDSegDataLayer'
    # else:
    #     pydata_params['voc_dir'] = '../../data/pascal/VOC2011'
    #     pylayer = 'VOCSegDataLayer'
    # n.data, n.label = L.Python(module='voc_layers', layer=pylayer,
    #         ntop=2, param_str=str(pydata_params))

    # input
    n.data = L.Input(input_param=dict(shape=dict(dim=input_dims)))
    if phase == 'train':
        n.label = L.Input(phase=0,input_param=dict(shape=dict(dim=input_dims)))

    # the base net
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64, pad=100, kernel_size=3, stride=1)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64, pad=1, kernel_size=3, stride=1)
    n.pool1 = max_pool(n.relu1_2, pad=0, kernel_size=2, stride=2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128, pad=1, kernel_size=3, stride=1)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128, pad=1, kernel_size=3, stride=1)
    n.pool2 = max_pool(n.relu2_2, pad=0, kernel_size=2, stride=2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256, pad=1, kernel_size=3, stride=1)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256, pad=1, kernel_size=3, stride=1)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256, pad=1, kernel_size=3, stride=1)
    n.pool3 = max_pool(n.relu3_3, pad=0, kernel_size=2, stride=2)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512, pad=1, kernel_size=3, stride=1)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512, pad=1, kernel_size=3, stride=1)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512, pad=1, kernel_size=3, stride=1)
    n.pool4 = max_pool(n.relu4_3, pad=0, kernel_size=2, stride=2)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512, pad=1, kernel_size=3, stride=1)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512, pad=1, kernel_size=3, stride=1)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512, pad=1, kernel_size=3, stride=1)
    n.pool5 = max_pool(n.relu5_3, pad=0, kernel_size=2, stride=2)

    # fully conv
    n.fc6, n.relu6 = conv_relu(n.pool5, 4096, pad=0, kernel_size=7, stride=1)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
    n.fc7, n.relu7 = conv_relu(n.drop6, 4096, pad=0, kernel_size=1, stride=1)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)
    # score
    n.score_fr = L.Convolution(n.drop7, num_output=num_class, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.upscore2 = L.Deconvolution(n.score_fr,
        convolution_param=dict(num_output=num_class, kernel_size=4, stride=2,
            bias_term=False),
        param=[dict(lr_mult=0)])

    n.score_pool4 = L.Convolution(n.pool4, num_output=num_class, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.score_pool4c = crop(n.score_pool4, n.upscore2)
    n.fuse_pool4 = L.Eltwise(n.upscore2, n.score_pool4c,
            operation=P.Eltwise.SUM)
    n.upscore16 = L.Deconvolution(n.fuse_pool4,
        convolution_param=dict(num_output=num_class, kernel_size=32, stride=16,
            bias_term=False),
        param=[dict(lr_mult=0)])

    n.score = crop(n.upscore16, n.data)
    if phase == 'train':
        n.loss = L.SoftmaxWithLoss(n.score, n.label,
            loss_param=dict(normalize=False, ignore_label=ignore_label))

    return n.to_proto()

def fcn_8s(input_dims, num_class, ignore_label=255, phase="train"):
    n = caffe.NetSpec()
    # pydata_params = dict(split=split, mean=(104.00699, 116.66877, 122.67892),
    #         seed=1337)
    # if split == 'train':
    #     pydata_params['sbdd_dir'] = '../data/sbdd/dataset'
    #     pylayer = 'SBDDSegDataLayer'
    # else:
    #     pydata_params['voc_dir'] = '../data/pascal/VOC2011'
    #     pylayer = 'VOCSegDataLayer'
    # n.data, n.label = L.Python(module='voc_layers', layer=pylayer,
    #         ntop=2, param_str=str(pydata_params))

    # input
    n.data = L.Input(input_param=dict(shape=dict(dim=input_dims)))
    if phase == 'train':
        n.label = L.Input(phase=0,input_param=dict(shape=dict(dim=input_dims)))

    # the base net
    n.conv1_1, n.relu1_1 = conv_relu(n.data, 64, pad=100, kernel_size=3, stride=1)
    n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64, pad=1, kernel_size=3, stride=1)
    n.pool1 = max_pool(n.relu1_2, pad=0, kernel_size=2, stride=2)

    n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128, pad=1, kernel_size=3, stride=1)
    n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128, pad=1, kernel_size=3, stride=1)
    n.pool2 = max_pool(n.relu2_2, pad=0, kernel_size=2, stride=2)

    n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256, pad=1, kernel_size=3, stride=1)
    n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256, pad=1, kernel_size=3, stride=1)
    n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256, pad=1, kernel_size=3, stride=1)
    n.pool3 = max_pool(n.relu3_3, pad=0, kernel_size=2, stride=2)

    n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512, pad=1, kernel_size=3, stride=1)
    n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512, pad=1, kernel_size=3, stride=1)
    n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512, pad=1, kernel_size=3, stride=1)
    n.pool4 = max_pool(n.relu4_3, pad=0, kernel_size=2, stride=2)

    n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512, pad=1, kernel_size=3, stride=1)
    n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512, pad=1, kernel_size=3, stride=1)
    n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512, pad=1, kernel_size=3, stride=1)
    n.pool5 = max_pool(n.relu5_3, pad=0, kernel_size=2, stride=2)

    # fully conv
    n.fc6, n.relu6 = conv_relu(n.pool5, 4096, pad=0, kernel_size=7, stride=1)
    n.drop6 = L.Dropout(n.relu6, dropout_ratio=0.5, in_place=True)
    n.fc7, n.relu7 = conv_relu(n.drop6, 4096, pad=0, kernel_size=1, stride=1)
    n.drop7 = L.Dropout(n.relu7, dropout_ratio=0.5, in_place=True)
    # score
    n.score_fr = L.Convolution(n.drop7, num_output=num_class, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.upscore2 = L.Deconvolution(n.score_fr,
        convolution_param=dict(num_output=num_class, kernel_size=4, stride=2,
            bias_term=False),
        param=[dict(lr_mult=0)])

    n.score_pool4 = L.Convolution(n.pool4, num_output=num_class, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.score_pool4c = crop(n.score_pool4, n.upscore2)
    n.fuse_pool4 = L.Eltwise(n.upscore2, n.score_pool4c,
            operation=P.Eltwise.SUM)
    n.upscore_pool4 = L.Deconvolution(n.fuse_pool4,
        convolution_param=dict(num_output=num_class, kernel_size=4, stride=2,
            bias_term=False),
        param=[dict(lr_mult=0)])

    n.score_pool3 = L.Convolution(n.pool3, num_output=num_class, kernel_size=1, pad=0,
        param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)])
    n.score_pool3c = crop(n.score_pool3, n.upscore_pool4)
    n.fuse_pool3 = L.Eltwise(n.upscore_pool4, n.score_pool3c,
            operation=P.Eltwise.SUM)
    n.upscore8 = L.Deconvolution(n.fuse_pool3,
        convolution_param=dict(num_output=num_class, kernel_size=16, stride=8,
            bias_term=False),
        param=[dict(lr_mult=0)])

    n.score = crop(n.upscore8, n.data)
    if phase == 'train':
        n.loss = L.SoftmaxWithLoss(n.score, n.label,
            loss_param=dict(normalize=False, ignore_label=ignore_label))

    return n.to_proto()

def make_fcn(net, input_dims, num_class, prototxt_train, prototxt_test):
    # register net
    __nets = ['fcn_32s', 'fcn_16s', 'fcn_8s']
    assert net in __nets, 'Unknown net: {}'.format(net)
    global ignore_label
    ignore_label = 255

    if net == 'fcn_32s':
        train_net = fcn_32s(input_dims, num_class, ignore_label, phase="train")
        test_net = fcn_32s(input_dims, num_class, ignore_label, phase="test")
    if net == 'fcn_16s':
        train_net = fcn_16s(input_dims, num_class, ignore_label, phase="train")
        test_net = fcn_16s(input_dims, num_class, ignore_label, phase="test")
    if net == 'fcn_8s':
        train_net = fcn_8s(input_dims, num_class, ignore_label, phase="train")
        test_net = fcn_8s(input_dims, num_class, ignore_label, phase="test")

    with open(prototxt_train, 'w') as f:
        f.write(str(train_net))

    with open(prototxt_test, 'w') as f:
        f.write(str(test_net))