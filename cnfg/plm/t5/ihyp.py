#encoding: utf-8

from cnfg.ihyp import *
from cnfg.plm.t5.hyp import *

# biases
enable_prev_ln_bias_default = enable_proj_bias_default = not ease_optimization

# computation order
norm_residual_default = not (computation_order.lower() == "v2")

# activation fucntion
use_adv_act_default = advance_activation_function is not None
adv_act = advance_activation_function.lower() if use_adv_act_default else None
inplace_after_Custom_Act = use_adv_act_default and (adv_act not in set(["sigmoid"]))

# relative position encoding
use_k_relative_position_encoder, use_k_relative_position_decoder = parse_double_value_tuple(use_k_relative_position)
rel_pos_enabled = (max(use_k_relative_position_encoder, use_k_relative_position_decoder) > 0)
relative_position_max_bucket_distance_encoder, relative_position_max_bucket_distance_decoder = parse_double_value_tuple(relative_position_max_bucket_distance)
relative_position_max_bucket_distance_cattn = relative_position_max_bucket_distance_encoder if use_k_relative_position_cattn > 0 else 0
disable_std_pemb_encoder, disable_std_pemb_decoder = parse_double_value_tuple(disable_std_pemb)
relpos_reduction_with_zeros = True

ieps_ln_default = 1e-06
