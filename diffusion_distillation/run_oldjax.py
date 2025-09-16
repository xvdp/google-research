

import os
import time
import requests
import functools
import jax
from jax import config
import jax.numpy as jnp
import flax
from matplotlib import pyplot as plt
import numpy as onp
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
from diffusion_distillation import diffusion_distillation



# create model
config = diffusion_distillation.config.cifar_base.get_config()
# test trainig with adam, no weight decay
config.train.weight_decay=0
model = diffusion_distillation.model.Model(config)

# WRONG version of jax.  model.make_init_state() AttributeError: module 'jax' has no attribute 'tree'
# init params 
state = jax.device_get(model.make_init_state())
state = flax.jax_utils.replicate(state)


