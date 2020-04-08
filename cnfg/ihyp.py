#encoding: utf-8

# this file interprets hyper-parameters assigned in cnfg/hyp.py

from cnfg.hyp import *

from math import inf

from utils.fmt.base import parse_none, parse_double_value_tuple

if ease_optimization:
	enable_residual_bias_default = False
else:
	enable_residual_bias_default = True

use_adv_act_default = False
override_GeLU_Swish = False
override_GeLU_Sigmoid = False
if advance_activation_function is not None:
	use_adv_act_default = True
	_adv_act = advance_activation_function.lower()
	if _adv_act == "sigmoid":
		override_GeLU_Sigmoid = True
	elif _adv_act == "swish":
		override_GeLU_Swish = True
inplace_after_GeLU = use_adv_act_default and not override_GeLU_Sigmoid

norm_residual_default = not (computation_order.lower() == "v2")

# override by the GoogleLR in most case
init_lr = 1e-4

inf_default = inf

ieps_default = 1e-9
ieps_ln_default = 1e-6
ieps_adam_default = 1e-9

ieps_ln_default = parse_none(ieps_ln_default, ieps_default)
ieps_adam_default = parse_none(ieps_adam_default, ieps_default)

adam_betas_default = (0.9, 0.98,)

use_k_relative_position_encoder, use_k_relative_position_decoder = parse_double_value_tuple(use_k_relative_position)
disable_std_pemb_encoder, disable_std_pemb_decoder = parse_double_value_tuple(disable_std_pemb)

h5datawargs = {} if hdf5_data_compression is None else {"compression": hdf5_data_compression, "compression_opts": hdf5_data_compression_level, "shuffle":True}
h5modelwargs = {} if hdf5_model_compression is None else {"compression": hdf5_model_compression, "compression_opts": hdf5_model_compression_level, "shuffle":True}
h5zipargs = {"compression": "gzip", "compression_opts": 9, "shuffle":True}

list_key_func = str
