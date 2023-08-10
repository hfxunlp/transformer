#encoding: utf-8

from cnfg.base import *

# new configurations for T5
model_name = ("encoder", "decoder",)
num_type = None
remove_classifier_bias = True
pre_trained_m = None

# override standard configurations
bindDecoderEmb = True
share_emb = True

isize = 768
ff_hsize = isize * 4
nhead = max(1, isize // 64)
attn_hsize = isize

nlayer = 12

drop = 0.1
attn_drop = drop
act_drop = drop

norm_output = True
