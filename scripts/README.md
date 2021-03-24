# Scripts

## `mktrain.sh`

Build hdf5 format cache for training data, this process including: sorting of the training data, building vocabulary, converting text to tensor and saving them into `.h5` files. Configure the following variables:


```
# the cache path of datasets
export cachedir=cache
# the ID of a dataset (files should be saved in $cachedir/$dataid)
export dataid=w14ende

# the path to source parallel text data
export srcd=$cachedir/$dataid
# the training file of the source language
export srctf=src.train.bpe
# the training file of the target language
export tgttf=tgt.train.bpe
# the validation file of the source language
export srcvf=src.dev.bpe
# the validation file of the target language
export tgtvf=tgt.dev.bpe

# the resulted tensor files for the training set and the development set
export rsf_train=train.h5
export rsf_dev=dev.h5

# using a shared vocabulary between the source language and the target language or not
export share_vcb=false
# "vsize" is the size of the vocabulary for both source language and its translation. Set a very large number to use the full vocabulary for BPE. The real vocabulary size will be 4 greater than this value because of special tags ("<sos>", "<eos>", "<unk>" and "<pad>").
export vsize=65536

# maximum number of tokens allowed for trained sentences
export maxtokens=256

# number of GPU(s) plan to use in training.
export ngpu=1

# sorting dataset and building vocabularies. true for the first time generation, false when only update the .h5 files.
export do_sort=true
export build_vocab=true
```

## `mktest.sh`

Performing translation of a given source text. Configure the following variables:

```
# "srcd" is the path of the source file you want to translate.
export srcd=w14src
# "srctf" is a plain text file to be translated which should be saved in "srcd" and processed with bpe like that with the training set.
export srctf=src-val.bpe

# the model file to perform the translation.
export modelf=expm/debug/checkpoint.t7

# path to save the translation result
export rsd=w14trs
# result file.
export rsf=trans.txt

# used a shared vocabulary while generating the training data?
export share_vcb=false

# the ID of the dataset assigned in scripts/mktrain.sh
export dataid=w14ende

# number of GPU(s) plan to use for decoding.
export ngpu=1

# merge sub-words
export debpe=true
```

## `bpe/`

Scripts to perform sub-word segmentation.


## `doc/`

Corresponding scripts for document-level data processing.

## `ape/`

Scripts for data processing of APE.
