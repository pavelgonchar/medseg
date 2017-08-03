import _init_paths
import os
import os.path as osp
from lits.config import cfg
from deeplab import make_deeplab
from fcn import make_fcn
from unet import make_unet
from resnet import make_resnet
from vnet import make_vnet
from densenet import make_densenet
from uvnet import make_uvnet

# register net
__backbones = ['deeplab', 'fcn', 'unet','resnet', 'vnet', 'densenet', 'uvnet']
__nets = ['deeplab_largefov', 'deeplab_largefov_bn', 'deeplab_v2_vgg_16', 'deeplab_v2_vgg_16_bn', 'deeplab_v2_vgg_16_unet',
		'fcn_32s', 'fcn_16s', 'fcn_8s',
		'unet_2d', 'unet_2d_bn', 'unet_3d', 'unet_3d_bn', 'unet_rcu_2d', 'unet_2d_bn_8s', 'unet_rcu_2d_bn', 'unet_2d_bn_mask', 'unet_2d_bn_weigted',
		'resnet_50', 'resnet_101', 'resnet_152',
		'vnet_2d', 'vnet_2d_bn',
		'densenet_unet', 'densenet_unet_8s',
		'uvnet_2d_bn', 'uvnet_2d_bn_weigted', 'uvnet_2d_bn_original_weigted', 'uvnet_2d_bn_modified_weigted', 'uvnet_2d_bn_incept_weigted', 'uvnet_2d_bn_incept2_weigted']
### setting ###
# dim_data = [5,1,156,156,8]
# dim_label = [5,1,156,156,8]
dim_data = [2,5,416,416]
dim_label = [2,1,416,416]
num_class = 2
net = 'uvnet_2d_bn_incept2_weigted'

backbone = net.split('_')[0]
assert backbone in __backbones, 'Unknown backbone: {}'.format(backbone)
assert net in __nets, 'Unknown net: {}'.format(net)

dirname = '{}_c{}'.format(net, num_class)
prototxt_dir = osp.abspath(osp.join(cfg.MODELS_DIR, backbone, dirname))
if not osp.exists(prototxt_dir):
	os.makedirs(prototxt_dir)
prototxt_train = osp.abspath(osp.join(prototxt_dir, 'train.prototxt'))
prototxt_test = osp.abspath(osp.join(prototxt_dir, 'test.prototxt'))

if net in ['deeplab_largefov', 'deeplab_largefov_bn', 'deeplab_v2_vgg_16', 'deeplab_v2_vgg_16_bn', 'deeplab_v2_vgg_16_unet']:
	make_deeplab(net, dim_data, dim_label, num_class, prototxt_train, prototxt_test)

if net in ['fcn_32s', 'fcn_16s', 'fcn_8s']:
	make_fcn(net, dim_data, dim_label, num_class, prototxt_train, prototxt_test)

if net in ['unet_2d', 'unet_2d_bn', 'unet_3d', 'unet_3d_bn', 'unet_rcu_2d', 'unet_2d_bn_8s', 'unet_rcu_2d_bn', 'unet_2d_bn_mask', 'unet_2d_bn_weigted']:
	make_unet(net, dim_data, dim_label, num_class, prototxt_train, prototxt_test)

if net in ['resnet_50', 'resnet_101', 'resnet_152']:
	make_resnet(net, dim_data, dim_label, num_class, prototxt_train, prototxt_test)

if net in ['vnet_2d', 'vnet_2d_bn']:
	make_vnet(net, dim_data, dim_label, num_class, prototxt_train, prototxt_test)

if net in ['densenet_unet', 'densenet_unet_8s']:
	make_densenet(net, dim_data, dim_label, num_class, prototxt_train, prototxt_test)

if net in ['uvnet_2d_bn', 'uvnet_2d_bn_weigted','uvnet_2d_bn_original_weigted', 'uvnet_2d_bn_modified_weigted', 'uvnet_2d_bn_incept_weigted', 'uvnet_2d_bn_incept2_weigted']:
	make_uvnet(net, dim_data, dim_label, num_class, prototxt_train, prototxt_test)