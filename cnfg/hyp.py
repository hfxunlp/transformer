#encoding: utf-8

ease_optimization = True

lipschitz_initialization = True

# choices: None, "GeLU", "Swish", "Sigmoid", "Mish", "NormSwish"
advance_activation_function = None

# choices: "v1", "v2"
computation_order = "v1"

# default cached sequence length (for positional embedding, etc.)
cache_len_default = 256

# window size (one side) of relative positional embeddings, 0 to disable. 8 and 16 are used in [Self-Attention with Relative Position Representations](https://www.aclweb.org/anthology/N18-2074/) for Transformer Base and Big respectively. disable_std_pemb to disable the standard positional embedding when use the relative position, or to disable only the decoder side with a tuple (False, True,), useful for AAN.
use_k_relative_position = 0
disable_std_pemb = False

# using fast implementation of label smoothing loss, but it cannot exclude the negative impact of special tokens, like <pad>, on training. `forbidden_indexes` in `cnfg/base.py` shall be set to None to enable.
use_fast_loss = False

# configure maximum batch size w.r.t GPU memory
max_sentences_gpu = 2048
max_tokens_gpu = 6144
max_pad_tokens_sentence = 32
normal_tokens_vs_pad_tokens = 4

# trade CPU for IO and disk space, see [h5py](http://docs.h5py.org/en/stable/high/dataset.html) for details.
# choices: None, "gzip", "lzf"
hdf5_data_compression = "gzip"
# choices: 0 to 9, default is 4. None for lzf.
hdf5_data_compression_level = 9
hdf5_model_compression = None
hdf5_model_compression_level = 0

# For BPE (using full vocabulary), the special <unk> token will never appear and thus can be removed from the vocabulary. Otherwise, it should be set to True.
use_unk = True

# prune with length penalty in each beam decoding step
clip_beam_with_lp = True
