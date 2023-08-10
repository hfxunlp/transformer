#encoding: utf-8

# this file interprets hyper-parameters assigned in cnfg/hyp.py

from math import inf

from utils.fmt.parser import parse_double_value_tuple, parse_none

from cnfg.hyp import *

# C backend
if use_c_backend is None:
	use_c_backend_attn = use_c_backend_selfattn = use_c_backend_crossattn = use_c_backend_pff = use_c_backend_group = True
	use_c_backend_act_func = False
else:
	use_c_backend_attn = use_c_backend_selfattn = use_c_backend_crossattn = use_c_backend_group = use_c_backend_pff = use_c_backend_act_func = use_c_backend
use_c_backend_mhattn = use_c_backend_attn or use_c_backend_selfattn or use_c_backend_crossattn
bind_c_forward = use_c_backend

# the use of deterministic algorithms
use_deterministic = not performance_over_reproduction
allow_tf32 = allow_fp16_reduction = performance_over_reproduction

# biases
enable_prev_ln_bias_default = enable_proj_bias_default = not ease_optimization

# computation order
norm_residual_default = not (computation_order.lower() == "v2")

# Layer Norm
enable_ln_parameters = True

# activation fucntion
use_adv_act_default = advance_activation_function is not None
adv_act = advance_activation_function.lower() if use_adv_act_default else None
inplace_after_Custom_Act = use_adv_act_default and (adv_act not in set(["sigmoid"]))

# relative position encoding
use_k_relative_position_encoder, use_k_relative_position_decoder = parse_double_value_tuple(use_k_relative_position)
rel_pos_enabled = (max(use_k_relative_position_encoder, use_k_relative_position_decoder) > 0)
relative_position_max_bucket_distance_encoder, relative_position_max_bucket_distance_decoder = parse_double_value_tuple(relative_position_max_bucket_distance)
disable_std_pemb_encoder, disable_std_pemb_decoder = parse_double_value_tuple(disable_std_pemb)
relpos_reduction_with_zeros = True

# torch_compile args
torch_compile_args = []
torch_compile_kwargs = {"fullgraph": False, "dynamic": False}

# hyper-parameters
inf_default = inf

ieps_default = 1e-9
ieps_ln_default = 1e-6
ieps_adam_default = 1e-9
ieps_ln_default = parse_none(ieps_ln_default, ieps_default)
ieps_adam_default = parse_none(ieps_adam_default, ieps_default)
ieps_noise_default = ieps_ln_default
ieps_upper_bound_default = ieps_default
ieps_dropout_multinomial_default = ieps_default

adam_betas_default = (0.9, 0.98,)

# HDF5 serialization
h5datawargs = {"track_times": False} if hdf5_data_compression is None else {"compression": hdf5_data_compression, "compression_opts": hdf5_data_compression_level, "shuffle": True, "track_times": False}
h5modelwargs = {"track_times": False} if hdf5_model_compression is None else {"compression": hdf5_model_compression, "compression_opts": hdf5_model_compression_level, "shuffle": True, "track_times": False}
h5zipargs = {"compression": "gzip", "compression_opts": 9, "shuffle":True, "track_times": False}
h5_libver = "latest" if hdf5_perf_over_camp else "earliest"

list_key_func = str

# tqdm
tqdm_mininterval = 1.0

# optimizer step zero_grad
optm_step_zero_grad_set_none = not contiguous_parameters
