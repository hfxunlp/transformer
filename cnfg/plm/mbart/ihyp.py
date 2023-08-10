#encoding: utf-8

from cnfg.ihyp import *
from cnfg.plm.mbart.hyp import *

# biases
enable_prev_ln_bias_default = enable_proj_bias_default = not ease_optimization

# computation order
norm_residual_default = not (computation_order.lower() == "v2")

# activation fucntion
use_adv_act_default = advance_activation_function is not None
adv_act = advance_activation_function.lower() if use_adv_act_default else None
inplace_after_Custom_Act = use_adv_act_default and (adv_act not in set(["sigmoid"]))

ieps_ln_default = 1e-05
