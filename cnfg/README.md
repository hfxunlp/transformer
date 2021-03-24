# cnfg

## `base.py`

Configure the following variables for training and translation:

```
# the group ID for the experiment
group_id = "std"

# an ID for your experiment. Model, log and state files will be saved in: expm/data_id/group_id/run_id
run_id = "base"

# the ID of the dataset to use
data_id = "w14ende"

# training, validation and test sets, created by mktrain.sh and mktest.sh correspondingly.
train_data = "cache/"+data_id+"/train.h5"
dev_data = "cache/"+data_id+"/dev.h5"
test_data = "cache/"+data_id+"/test.h5"

# the saved model file to fine tune with.
fine_tune_m = None

# non-exist indexes in the classifier.
# "<pad>":0, "<sos>":1, "<eos>":2, "<unk>":3
# add 3 to forbidden_indexes if there are <unk> tokens in data
forbidden_indexes = [0, 1]

# after how much step save a checkpoint which you can fine tune with.
save_every = 1500
# maximum number of checkpoint models saved, useful for average or ensemble.
num_checkpoint = 4
# start saving checkpoints only after this epoch
epoch_start_checkpoint_save = 3

# optimize after the number of trained tokens is larger than "tokens_optm", designed to support large batch size on a single GPU effectively.
tokens_optm = 25000

# number of continuous epochs where no smaller validation loss found to early stop the training.
earlystop = 8
# maximum training epochs.
maxrun = 128
# number of training steps, 300000 for transformer big.
training_steps = 100000

# report training loss after these many optimization steps, and whether report evaluation result or not.
batch_report = 2000
report_eva = False

# run on GPU or not, and GPU device(s) to use. Data Parallel multi-GPU support can be enabled with values like: 'cuda:0, 1, 3'. Set gpuid to None to use all GPUs.
use_cuda = True
gpuid = 'cuda:0'
# use mixed precision (FP16)
use_amp = False
# use multi-gpu optimizer, may help bring slight acceleration for the training of large models (e.g. deep/big Transformers) with complex optimizers (e.g. Adam).
multi_gpu_optimizer = True

# bind the embedding matrix with the classifer weight in decoder
bindDecoderEmb = True
# sharing embedding of the encoder and the decoder or not.
share_emb = False

# size of the embeddings.
isize = 512
# hidden size for those feed-forward neural networks.
ff_hsize = isize * 4
# number of heads for multi-head attention.
nhead = max(1, isize // 64)
# hidden size for the attention model.
attn_hsize = None

# number of layers for encoder and decoder.
nlayer = 6

# dropout rate for hidden states.
drop = 0.1
# dropout rate applied to multi-head attention.
attn_drop = drop

# False for Hier/Incept Models
norm_output = True

# warm up steps for the training.
warm_step = 8000
# scalar of learning rate
lr_scale = 1.0

# label smoothing settings for the KL divergence.
label_smoothing = 0.1

# L2 regularization, 1e-5 for not very large dataset from The Best of BothWorlds: Combining Recent Advances in Neural Machine Translation
weight_decay = 0

# beam size for generating translations. Decoding of batches of data is supported, but requires more memory. Set to 1 for greedy decoding.
beam_size = 4
# length penalty applied to translating
length_penalty = 0.0
# use multi-gpu for translating or not. "predict.py" will take the last gpu rather than the first in case multi_gpu_decoding is set to False to avoid potential break due to out of memory, because the first gpu is the main device by default which takes more jobs.
multi_gpu_decoding = False

# random seed
seed = 666666

# save a model for every epoch regardless whether a lower loss/error rate has been reached. Useful for ensemble.
epoch_save = True

# to accelerate training through sampling, 0.8 and 0.1 in: Dynamic Sentence Sampling for Efficient Training of Neural Machine Translation
dss_ws = None
dss_rm = None

# apply ams for adam or not.
use_ams = False

# load embeddings retrieved with tools/check/ext_emb.py, and whether update them or not
src_emb = None
freeze_srcemb = False
tgt_emb = None
freeze_tgtemb = False
# scale down loaded embedding by sqrt(isize) or not, True as default to make positional embedding meaningful at beginning.
scale_down_emb = True

# training state and optimizer state files to resume training.
train_statesf = None
fine_tune_state = None

# saving the optimizer state or not.
save_optm_state = False
# saving shuffled sequence of training set or not 
save_train_state = False
```

## `hyp.py`

Configuration of following variables:

```
# reducing the optimization difficulty of models
ease_optimization = True

# using lipschitz constraint parameter initialization
lipschitz_initialization = True

# using advanced activation function, choices: None, "GeLU", "Swish", "Sigmoid", "NormSwish"
advance_activation_function = None

# computation order in Transformer sub-layer choices: "v1", "v2"
computation_order = "v1"

# default cached sequence length (for positional embedding, etc.)
cache_len_default = 256

# window size (one side) of relative positional embeddings, 0 to disable. 16 and 8 are used in [Self-Attention with Relative Position Representations](https://www.aclweb.org/anthology/N18-2074/) for Transformer Base and Big respectively. disable_std_pemb to disable the standard positional embedding when use the relative position, or to disable only the decoder side with a tuple (False, True,), useful for AAN.
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
```

## `ihyp.py`

To interpret configurations in `hyp.py`.

## `dynb.py`

Additional configurations for dynamic batch sizes.

```
# If the angle change is greater than or equal to the minimum value in the history * dyn_tol_alpha, perform an optimization step.
dyn_tol_alpha = 1.1
# If fails to obtain a smaller angle change after this number of steps, perform an optimization step.
dyn_tol_amin = 3

# override the maximum tokens per batch configuration in `cnfg/base.py`. If there are no less than this number of tokens in a batch, an optimization step will be performed.
tokens_optm = tokens_optm * 10

# perform optimization step only in case the angle change is smaller than update_angle.
update_angle = 90.0 / dyn_tol_alpha

# number of records of the angle change reduction.
num_dynb_his = 50

# hyper parameter for parameter sampling. Ignored in case using softmax over normalized angle change reduction (default). Uncomment corresponding lines in `utils/dynbatch.py` to enable.
select_alpha = 3.0
```

## `docpara.py`

Additional configurations for context-aware models.

```
# number of previous context sentences utilized
num_prev_sent = 2

# freeze the loaded sentence-level model
freeze_load_model = True
# unfreeze the bias and the weight matrix of the classifier of the sentence-level model
unfreeze_bias = True
unfreeze_weight = False

# number of layers for context encoding
num_layer_context = 1
```

