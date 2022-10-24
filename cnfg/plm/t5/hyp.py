#encoding: utf-8

from cnfg.hyp import *

ease_optimization = True

# choices: None, "GeLU", "Swish", "Sigmoid", "SReLU", "Mish", "NormSwish"
advance_activation_function = None

# choices: "v1", "v2"
computation_order = "v2"

# default cached sequence length (for positional embedding, etc.)
cache_len_default = 512

# For BPE (using full vocabulary), the special <unk> token will never appear and thus can be removed from the vocabulary. Otherwise, it should be set to True.
use_unk = True

# window size (one side) of relative positional embeddings, 0 to disable. 8 and 16 are used in [Self-Attention with Relative Position Representations](https://www.aclweb.org/anthology/N18-2074/) for Transformer Base and Big respectively. relative_position_max_bucket_distance for the bucket relative positional encoding used by T5, [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://www.jmlr.org/papers/v21/20-074.html), which hampers the performance. disable_std_pemb to disable the standard positional embedding when use the relative position, or to disable only the decoder side with a tuple (False, True,), useful for AAN.
use_k_relative_position = (15, 31,)
relative_position_max_bucket_distance = 128
use_k_relative_position_cattn = 15
disable_std_pemb = True
