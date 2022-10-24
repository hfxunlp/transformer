#encoding: utf-8

ease_optimization = True

lipschitz_initialization = True

# choices: None, "GeLU", "Swish", "Sigmoid", "SReLU", "Mish", "NormSwish"
advance_activation_function = None
# choices: None, "GLU" or above activation functions.
use_glu_ffn = None

# choices: "v1", "v2"
computation_order = "v2"

# default cached sequence length (for positional embedding, etc.)
cache_len_default = 256

# window size (one side) of relative positional embeddings, 0 to disable. 8 and 16 are used in [Self-Attention with Relative Position Representations](https://www.aclweb.org/anthology/N18-2074/) for Transformer Base and Big respectively. relative_position_max_bucket_distance for the bucket relative positional encoding used by T5, [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://www.jmlr.org/papers/v21/20-074.html), which slightly hampers the performance on WMT 14 En-De. disable_std_pemb to disable the standard positional embedding when use the relative position, or to disable only the decoder side with a tuple (False, True,), useful for AAN.
use_k_relative_position = 0
relative_position_max_bucket_distance = 0
disable_std_pemb = False

# using fast implementation of label smoothing loss, but it cannot exclude the negative impact of special tokens, like <pad>, on training. `forbidden_indexes` in `cnfg/base.py` shall be set to None to enable.
use_fast_loss = True

# configure maximum batch size w.r.t GPU memory
max_tokens_gpu = 6144
max_sentences_gpu = max_tokens_gpu // 6
max_pad_tokens_sentence = 32
normal_tokens_vs_pad_tokens = 4

# For BPE (using full vocabulary), the special <unk> token will never appear and thus can be removed from the vocabulary. Otherwise, it should be set to True.
use_unk = True

# enable tqdm progress bar.
enable_tqdm = True

# trade CPU for IO and disk space, see [h5py](http://docs.h5py.org/en/stable/high/dataset.html) for details.
# choices: None, "gzip", "lzf"
hdf5_data_compression = "gzip"
# choices: 0 to 9, default is 4. None for lzf.
hdf5_data_compression_level = 9
hdf5_model_compression = None
hdf5_model_compression_level = 0
# using the latest HDF5 version for its advantages even this forgos compatibility, see [h5py.File](https://docs.h5py.org/en/stable/high/file.html#version-bounding) for details.
hdf5_perf_over_camp = True
# whether to track creation order.
hdf5_track_order = False

# prune with length penalty in each beam decoding step
clip_beam_with_lp = True

# optimize speed even if it sacrifices reproduction
performance_over_reproduction = True

# enable torch checks, only support anomaly detection for the autograd engine currently.
enable_torch_check = True

# accelerate optimizer by using contigous parameters and gradients. Disabling it leads to better performance.
contiguous_parameters = False

# the number of checkpoints kept for `cnfg.save_auto_clean`
n_keep_best = 1

# use C backend. Disabling it leads to better performance.
use_c_backend = False
