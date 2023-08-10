#encoding: utf-8

from cnfg.hyp import *

ease_optimization = False

# choices: None, "GeLU", "Swish", "Sigmoid", "SReLU", "Mish", "NormSwish"
advance_activation_function = "GeLU"

# choices: "v1", "v2"
computation_order = "v1"

# default cached sequence length (for positional embedding, etc.)
cache_len_default = 1026

# For BPE (using full vocabulary), the special <unk> token will never appear and thus can be removed from the vocabulary. Otherwise, it should be set to True.
use_unk = True

# learning rate
init_lr = 1e-5
