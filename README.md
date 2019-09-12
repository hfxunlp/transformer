# Neutron
Neutron: A pytorch based implementation of [Transformer](https://arxiv.org/abs/1706.03762) and its variants.

This project is developed with python 3.7.

## Setup dependencies

Try `pip install -r requirements.txt` after you clone the repository.

If you want to use [BPE](https://github.com/rsennrich/subword-nmt), to enable convertion to C libraries, to try the simple MT server and to support Chinese word segmentation supported by [pynlpir](https://github.com/tsroten/pynlpir) in this implementation, you should also install those dependencies in `requirements.opt.txt` with `pip install -r requirements.opt.txt`.

## Data preprocessing

### BPE

We provide scripts to apply Byte-Pair Encoding (BPE) under `scripts/bpe/`.

### convert plain text to tensors for training

Generate training data for `train.py` with `bash scripts/mktrain.sh`, configure following variables in `mktrain.sh` for your usage (the other variables should comply with those in `scripts/mkbpe.sh`):

```
# the path of datasets
export cachedir=cache
# the ID of a dataset (files should be saved in $cachedir/$dataid)
export dataid=w14ende
# the training file of the source language
export srctf=src.train.bpe
# the training file of the target language
export tgttf=tgt.train.bpe
# the validation file of the source language
export srcvf=src.dev.bpe
# the validation file of the target language
export tgtvf=tgt.dev.bpe

# "vsize" is the size of the vocabulary for both source language and its translation. Set a very large number to use the full vocabulary for BPE. The real vocabulary size will be 4 greater than this value because of special tags ("<sos>", "<eos>", "<unk>" and "<pad>").
export vsize=65536

# maximum number of tokens allowed for trained sentences
export maxtokens=256

# number of GPU(s) plan to use in training.
export ngpu=1
```

## Configuration for training and testing

All parameters for configuration are saved in `cnfg.py`:

```
# an ID for your experiment. Model, log and state files will be saved in: expm/data_id/run_id
run_id = "base"

# the ID of the dataset to use
data_id = "w14ende"

# training, validation and test sets, created by mktrain.sh and mktest.sh correspondingly.
train_data = "cache/"+data_id+"/train.h5"
dev_data = "cache/"+data_id+"/dev.h5"
test_data = "cache/"+data_id+"/test.h5"

# non-exist indexes in the classifier.
# "<pad>":0, "<sos>":1, "<eos>":2, "<unk>":3
# add 3 to forbidden_indexes if there are <unk> tokens in data
forbidden_indexes = [0, 1]

# the saved model file to fine tune with.
fine_tune_m = None
# corresponding training states files.
train_statesf = None
fine_tune_state = None

# load embeddings retrieved with tools/check/ext_emb.py, and whether update them or not
src_emb = None
freeze_srcemb = False
tgt_emb = None
freeze_tgtemb = False
# scale down loaded embedding by sqrt(isize) or not, True as default to make positional embedding meaningful at beginning.
scale_down_emb = True

# saving the optimizer state or not.
save_optm_state = False
# saving shuffled sequence of training set or not 
save_train_state = False

# after how much step save a checkpoint which you can fine tune with.
save_every = 1500
# maximum number of checkpoint models saved, useful for average or ensemble.
num_checkpoint = 4
# start saving checkpoints only after this epoch
epoch_start_checkpoint_save = 3

# save a model for every epoch regardless whether a lower loss/error rate has been reached. Useful for ensemble.
epoch_save = True

# beam size for generating translations. Decoding of batches of data is supported, but requires more memory. Set to 1 for greedy decoding.
beam_size = 4

# number of continuous epochs where no smaller validation loss found to early stop the training.
earlystop = 8

# maximum training epochs.
maxrun = 128

# optimize after the number of trained tokens is larger than "tokens_optm", designed to support large batch size on a single GPU effectively.
tokens_optm = 25000

# report training loss after these many optimization steps, and whether report evaluation result or not.
batch_report = 2000
report_eva = False

# run on GPU or not, and GPU device(s) to use. Data Parallel depended multi-gpu support can be enabled with values like: 'cuda:0, 1, 3'.
use_cuda = True
gpuid = 'cuda:0'
# [EXP] enable mixed precision (FP16) with "O1"
amp_opt = None

# use multi-gpu for translating or not. "predict.py" will take the last gpu rather than the first in case multi_gpu_decoding is set to False to avoid potential break due to out of memory, because the first gpu is the main device by default which takes more jobs.
multi_gpu_decoding = False

# number of training steps, 300000 for transformer big.
training_steps = 100000

# to accelerate training through sampling, 0.8 and 0.1 in: Dynamic Sentence Sampling for Efficient Training of Neural Machine Translation
dss_ws = None
dss_rm = None

# apply ams for adam or not.
use_ams = False

# bind the embedding matrix with the classifer weight in decoder
bindDecoderEmb = True

# False for Hier/Incept Models
norm_output = True

# size of the embeddings.
isize = 512

# number of layers for encoder and decoder.
nlayer = 6

# hidden size for those feed-forward neural networks.
ff_hsize = isize * 4

# dropout rate for hidden states.
drop = 0.1

# dropout rate applied to multi-head attention.
attn_drop = drop

# label smoothing settings for the KL divergence.
label_smoothing = 0.1

# L2 regularization, 1e-5 for not very large dataset from The Best of BothWorlds: Combining Recent Advances in Neural Machine Translation
weight_decay = 0

# length penalty applied to translating
length_penalty = 0.0

# sharing embedding of the encoder and the decoder or not.
share_emb = False

# number of heads for multi-head attention.
nhead = max(1, isize // 64)

# maximum steps cached for the positional embedding.
cache_len = 256

# warm up steps for the training.
warm_step = 8000

# hidden size for the attention model.
attn_hsize = None

# random seed
seed = 666666
```

## Training

Just execute the following command to launch the training:

`python train.py (runid)`

where `runid` can be omitted. In that case, the `run_id` in `cnfg.py` will be taken as the id of the experiment.

## Generation

`bash scripts/mktest.sh`, configure following variables in `mktest.sh` for your usage (while keep the other settings consistent with those in `scripts/mkbpe.sh` and `scripts/mktrain.sh`):

```
# "srcd" is the path of the source file you want to translate.
export srcd=w14src

# "srctf" is a plain text file to be translated which should be saved in "srcd" and processed with bpe like that with the training set.
export srctf=src-val.bpe
# the model file to complete the task.
export modelf=expm/debug/checkpoint.t7
# result file.
export rsf=trans.txt

# the ID of the dataset assigned in mktrain.sh
export dataid=w14ende
```

## Exporting python files to C libraries

You can convert python classes into C libraries with `python mkcy.py build_ext --inplace`, and codes will be checked before compiling, which can serve as a simple to way to find typo and bugs as well. This function is supported by [Cython](https://cython.org/). These files can be removed with `rm -fr *.c *.so parallel/*.c parallel/*.so transformer/*.c transformer/*.so  transformer/AGG/*.c transformer/AGG/*.so build/`. Loading modules from compiled C libraries may also accelerate, but not significantly.

## Ranking

You can rank your corpus with pre-trained model, per token perplexity will be given for each sequence pair. Use it with:

`python rank.py rsf h5f models`

where `rsf` is the result file, `h5f` is HDF5 formatted input of file of your corpus (genrated like training set with `tools/mkiodata.py` like in `scripts/mktrain.sh`), `models` is a (list of) model file(s) to make perplexity evaluation.

## The other files' discription

### `modules/`

Foundamental models needed for the construction of transformer.

### `loss.py`

Implementation of label smoothing loss function required by the training of transformer.

### `lrsch.py`

Learning rate schedule model needed according to the paper.

### `utils.py`

Functions for basic features, for example, freeze / unfreeze parameters of models, padding list of tensors to same size on assigned dimension.

### `translator.py`

Provide an encapsulation for the whole translation procedure with which you can use the trained model in your application easier.

### `server.py`

An example depends on Flask to provide simple Web service and REST API about how to use the `translator`, configure [those variables](https://github.com/anoidgit/transformer/blob/master/server.py#L13-L21) before you use it.

### `transformer/`

#### `NMT.py`

The transformer model encapsulates encoder and decoder. Switch [the comment line](https://github.com/anoidgit/transformer/blob/master/transformer/NMT.py#L9-L11) to make a choice between the standard decoder and the average decoder.

#### `Encoder.py`

The encoder of transformer.

#### `Decoder.py`

The standard decoder of transformer.

#### `AvgDecoder.py`

The average decoder of transformer proposed by [Accelerating Neural Transformer via an Average Attention Network](https://arxiv.org/abs/1805.00631).

#### `EnsembleNMT.py`

A model encapsulates several NMT models to do ensemble decoding. Switch [the comment line](https://github.com/anoidgit/transformer/blob/master/transformer/EnsembleNMT.py#L9-L11) to make a choice between the standard decoder and the average decoder.

#### `EnsembleEncoder.py`

A model encapsulates several encoders for ensemble decoding.

#### `EnsembleDecoder.py`

A model encapsulates several standard decoders for ensemble decoding.

#### `EnsembleAvgDecoder.py`

A model encapsulates several average decoders proposed by [Accelerating Neural Transformer via an Average Attention Network](https://arxiv.org/abs/1805.00631) for ensemble decoding.

#### `AGG/`

Implementation of aggregation models.

##### `Hier*.py`

Hierarchical aggregation proposed in [Exploiting Deep Representations for Neural Machine Translation](https://arxiv.org/abs/1810.10181).

#### `TA/`

Implementation of transparent attention proposed in [Training Deeper Neural Machine Translation Models with Transparent Attention](https://aclweb.org/anthology/D18-1338) modules.

#### `SC/`

Implementation of sentential context proposed in [Exploiting Sentential Context for Neural Machine Translation](https://www.aclweb.org/anthology/P19-1624/) modules.

### `parallel/`

#### `base.py`

Implementations of `DataParallelModel` and `DataParallelCriterion` which support effective multi-GPU training and evaluation.

#### `parallelMT.py`

Implementation of `DataParallelMT` which supports paralleled decoding over multiple GPUs. 

### `datautils/`

#### `bpe.py`

A tool borrowed from [subword-nmt](https://github.com/rsennrich/subword-nmt) to apply bpe for translator

#### `moses.py`

Codes to encapsulate moses scripts, you have to define `moses_scripts`(path to moses scripts) and ensure `perl` is executable to use it, otherwise, you need to modify [these two lines](https://github.com/anoidgit/transformer/blob/master/datautils/moses.py#L7-L8) to tell the module where to find them.

#### `zh.py`

Chinese segmentation is different from tokenization, a tool is provided to support Chinese based on [pynlpir](https://github.com/tsroten/pynlpir).

### `tools/`

#### `average_model.py`

A tool to average several models to one which may bring some additional performance with no additional costs. Example usage:

`python tools/average_model.py $averaged_model_file.t7 $model1.t7 $model2.t7 ...`

#### `sort.py`

Sort the dataset to make the training more easier and start from easier questions.

#### `vocab.py`

Build vocabulary for the training set.

#### `mkiodata.py`

Convert text data to hdf5 format for the training script. Settings for the training data like batch size, maximum tokens per batch unit and padding limitation can be found [here](https://github.com/anoidgit/transformer/blob/master/tools/mkiodata.py#L157).

#### `mktest.py`

Convert translation requests to hdf5 format for the prediction script. Settings for the test data like batch size, maximum tokens per batch unit and padding limitation can be found [here](https://github.com/anoidgit/transformer/blob/master/tools/mktest.py#L140).

#### `lsort/`

Scripts to support sorting very large training set with limited memory.

#### `check/`

##### `debug/`
Tools to check the implementation and the data.

##### `fbindexes.py`

When you using a shared vocabulary for source side and target side, there are still some words which only appear at the source side even joint BPE is applied. Those words take up probabilities in the label smoothing classifier, and this tool can prevent this through generating a larger and well covered forbidden indexes list which can be concatnated to `forbidden_indexes` in `cnfg.py`.

#### `clean/`

Tools to filter the datasets.

## Performance

Speed on nVidia TITAN X GPU(s) measured by decoding tokens (`<eos>` counted and `<pad>` discounted) per second during training:

| Number of GPU(s) | ~ tokens/s |
| :------| ------: |
| 1 | 5800 |
| 2 | 10100 |

1, Settings: [WMT 2017, German -> English task](http://data.statmt.org/wmt17/translation-task/preprocessed/de-en/).

| Options | Value |
| :------| ------: |
| Vocabulary size (special tokens included) | 20130/23476 |
| Maximum sentence length | 256 |
| Embedding size | 512 |
| Hidden units of Feed-forward neural network | 2048 |
| Encoder/Decoder layers | 6 |
| Minimum decoded tokens per optimization step | 32768 |
| Number of heads in Multi-head Attention | 8 |
| Dropout in Feed-forward neural network | 0.1 |
| Dropout in Multi-head Attention | 0.1 |
| Label smoothing | 0.1 |
| Beam size | 4 |

Measured with `multi-bleu-detok.perl`:

| Newstest 2017 | THUMT | Epoch 26 | Epoch 72 | Epoch 72 (Length Penalty: 0.6) |
| :------| ------: | ------: | ------: | ------: |
| Case-sensitive | 32.63 | 32.26 | 32.97 | 32.89 |
| Case-insensitive | 34.06 | 33.70 | 34.36 | 34.28 |

Note: The result of [THUMT implementation](https://github.com/thumt/THUMT) is from [Accelerating Neural Transformer via an Average Attention Network](https://arxiv.org/abs/1805.00631). Averaging of models is not applied in the test of this implementation, since when layer normalization is applied between residue connections, averaging model might hurts performance. Results with length penalty as THUMT applied is reported, but length penalty does not improve the performance of transformer in my experiments. Outputs of last encoder layer and decoder layer are not normalised in this experiment.

2, Settings: same with the first except the outputs of last encoder layer and decoder layer are normed and:

| Options | Value |
| :------| ------: |
| Vocabulary size (special tokens included) | 19667/23006 |
| Minimum decoded tokens per optimization step | 25000 |

Measured with `multi-bleu-detok.perl`:

| Newstest 2017 | Baseline 20 | Baseline 21 | Baseline 43 | Avg Attn 25 | GeLU 17 | SparseNorm 17 | Iterative DeepR 19 | Hierarchical DeepR 25 | Hierarchical 0.1 27 | Incept 25 | EncEmb Noise 21 |
| :------| ------: | ------: | ------: | ------: | ------: | ------: | ------: | ------: | ------: | ------: | ------: |
| Case-sensitive | 32.03 | 32.09 | 32.60 | 31.93 | 32.13 | 32.17 | 32.07 | 32.44 | 32.55 | 32.87 | 32.29 |
| Case-insensitive | 33.46 | 33.55 | 34.06 | 33.33 | 33.53 | 33.63 | 33.47 | 33.89 | 33.93 | 34.32 | 33.73 |

3, Settings: WMT 2014, English -> German, 32k independent BPE with 8 as vocabulary threshold for BPE.

| Options | Value |
| :------| ------: |
| Vocabulary size (special tokens included) | 34324/34429 |
| Minimum decoded tokens per optimization step | 25000 |

Measured with `multi-bleu-detok.perl`:

| Newstest 2014 | Original Paper | Averaging 5 Checkpoints for Original Paper | Original Implementation | Averaging 5 Checkpoints for Original Implementation |
| :------| ------: | ------: | ------: | ------: |
| Case-sensitive | 27.41 | 27.77 | 27.18  | 27.41 |
| Case-insensitive | 27.95 | 28.32 | 27.72  | 27.96 |

There is a difference between the Transformer in the original paper (residue connections are layer normalised) and their official implementation (not normalised).

## Acknowledgements

The project starts when Hongfei XU (the developer) was a postgraduate student at [Zhengzhou University](http://www5.zzu.edu.cn/nlp/), and continues when he is a PhD candidate at [Saarland University](https://www.uni-saarland.de/nc/en/home.html) supervised by [Prof. Dr. Josef van Genabith](https://www.dfki.de/en/web/about-us/employee/person/jova02/) and Prof. Dr. Deyi Xiong, and a Junior Researcher at [DFKI, MLT (German Research Center for Artificial Intelligence, Multilinguality and Language Technology)](https://www.dfki.de/en/web/research/research-departments-and-groups/multilinguality-and-language-technology/). Hongfei XU enjoys a doctoral grant from [China Scholarship Council](https://www.csc.edu.cn/) ([2018]3101, 201807040056) while maintaining this project.

Details of this project can be found [here](https://arxiv.org/abs/1903.07402), and please cite it if you enjoy the implementation :)

```
@article{xu2019neutron,
  author          = {Xu, Hongfei and Liu, Qiuhui},
  title           = "{Neutron: An Implementation of the Transformer Translation Model and its Variants}",
  journal         = {arXiv preprint arXiv:1903.07402},
  archivePrefix   = "arXiv",
  eprinttype      = {arxiv},
  eprint          = {1903.07402},
  primaryClass    = "cs.CL",
  keywords        = {Computer Science - Computation and Language},
  year            = 2019,
  month           = "March",
  url             = {https://arxiv.org/abs/1903.07402},
  pdf             = {https://arxiv.org/pdf/1903.07402}
}
```

## Contributor(s)

## Need more?

Every details are in those codes, just explore them and make commits ;-)
