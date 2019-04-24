#encoding: utf-8

run_id = "base"

data_id = "de-en"

train_data = "cache/"+data_id+"/train.h5"
dev_data = "cache/"+data_id+"/dev.h5"
test_data = "cache/"+data_id+"/test.h5"

# non-exist indexes in the classifier.
# "<pad>":0, "<sos>":1, "<eos>":2, "<unk>":3
# add 3 to forbidden_indexes if there are <unk> tokens in data
forbidden_indexes = [0, 1]

fine_tune_m = None
train_statesf = None
fine_tune_state = None

src_emb = None
freeze_srcemb = False
tgt_emb = None
freeze_tgtemb = False
scale_down_emb = True

save_optm_state = False
save_train_state = False

save_every = 1500
num_checkpoint = 4
epoch_start_checkpoint_save = 3

epoch_save = True

beam_size = 4

earlystop = 8

maxrun = 128

tokens_optm = 25000

batch_report = 5000
report_eva = False

use_cuda = True
# enable Data Parallel multi-gpu support with values like: 'cuda:0, 1, 3'.
gpuid = 'cuda:0'

# use multi-gpu for translating or not. `predict.py` will take the last gpu rather than the first in case multi_gpu_decoding is set to False to avoid potential break due to out of memory, since the first gpu is the main device by default which takes more jobs.
multi_gpu_decoding = False

training_steps = 100000

# to accelerate training through sampling, 0.8 and 0.1 in: Dynamic Sentence Sampling for Efficient Training of Neural Machine Translation
dss_ws = None
dss_rm = None

use_ams = False

bindDecoderEmb = True

# False for Hier/Incept Models
norm_output = True

isize = 512

nlayer = 6

ff_hsize = 2048

drop = 0.1

attn_drop = 0.1

label_smoothing = 0.1

weight_decay = 0

length_penalty = 0.0

share_emb = False

nhead = 8

cache_len = 260

warm_step = 8000

attn_hsize = None

seed = 666666
