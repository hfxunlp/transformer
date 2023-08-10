#encoding: utf-8

from cnfg.base import *

# new configurations for RoBERTa
model_name = "roberta"
num_type = 1
eliminate_type_emb = False
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
act_drop = 0.0

norm_output = True
