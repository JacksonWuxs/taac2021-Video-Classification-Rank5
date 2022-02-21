import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2
import functools
import tensorflow.contrib.slim as slim

import src.model.image_head.efficientNet.efficientnet_builder as efficientnet_builder

networks_map={
	'resnet_v2_50': resnet_v2.resnet_v2_50,
	'resnet_v2_101': resnet_v2.resnet_v2_101,
	'resnet_v2_152': resnet_v2.resnet_v2_152,
	'resnet_v2_200': resnet_v2.resnet_v2_200,
        'efficientnet': efficientnet_builder.build_model_base,
}

arg_scopes_map={
	'resnet_v2_50': resnet_v2.resnet_arg_scope,
	'resnet_v2_101': resnet_v2.resnet_arg_scope,
	'resnet_v2_152': resnet_v2.resnet_arg_scope,
	'resnet_v2_200': resnet_v2.resnet_arg_scope,
}

def get_network_fn(name, model_name = None):
  if name not in networks_map:
    raise ValueError('Name of network unknown %s' % name)
  func = networks_map.get(name, None)
  if model_name is not None:
      func = functools.partial(func, model_name = model_name)
  @functools.wraps(func)
  def network_fn(images, is_training, **kwargs):
    if arg_scopes_map.get(name,None) is not None:
        arg_scope = arg_scopes_map[name](weight_decay=1e-5)
        with slim.arg_scope(arg_scope):
          out, _ = func(images, num_classes=None, is_training=is_training, **kwargs)
    else:
        out, _ = func(images, is_training=is_training, **kwargs)
    if len(out.get_shape()) ==4:
      return out[:,0,0,:] #squeeze conv feat
    else:
      return out

  if hasattr(func, 'default_image_size'):
    network_fn.default_image_size = func.default_image_size

  return network_fn

def get_instance(name, paramters):
    return get_network_fn(name, **paramters)
