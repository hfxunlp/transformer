# Neutron
Neutron: A pytorch based implementation of the [Transformer](https://arxiv.org/abs/1706.03762) and its variants.

This project is developed with python 3.10.

## Setup dependencies

Try `pip install -r requirements.txt` after you clone the repository.

If you want to use [BPE](https://github.com/rsennrich/subword-nmt), to enable convertion to C libraries, to try the simple MT server and to support Chinese word segmentation supported by [pynlpir](https://github.com/tsroten/pynlpir) in this implementation, you should also install those dependencies in `requirements.opt.txt` with `pip install -r requirements.opt.txt`.

## Data preprocessing

### BPE

We provide scripts to apply Byte-Pair Encoding (BPE) under `scripts/bpe/`.

### convert plain text to tensors for training

Generate training data for `train.py` with `bash scripts/mktrain.sh`, [configure variables](scripts/README.md#mktrainsh) in `scripts/mktrain.sh` for your usage (the other variables shall comply with those in `scripts/bpe/mk.sh`).

## Configuration for training and testing

Most [configurations](cnfg/README.md#basepy) are managed in `cnfg/base.py`. [Configure advanced details](cnfg/README.md#hyppy) with `cnfg/hyp.py`.

## Training

Just execute the following command to launch the training:

`python train.py`

## Generation

`bash scripts/mktest.sh`, [configure variables](scripts/README.md#mktestsh) in `scripts/mktest.sh` for your usage (while keep the other settings consistent with those in `scripts/mkbpe.sh` and `scripts/mktrain.sh`).

## Exporting python files to C libraries

You can convert python classes into C libraries with `python mkcy.py build_ext --inplace`, and codes will be checked before compiling, which can serve as a simple to way to find typo and bugs as well. This function is supported by [Cython](https://cython.org/). These files can be removed by commands `tools/clean/cython.py .` and `rm -fr build/`. Loading modules from compiled C libraries may also accelerate, but not significantly.

## Ranking

You can rank your corpus with pre-trained model, per token perplexity will be given for each sequence pair. Use it with:

`python rank.py rsf h5f models`

where `rsf` is the result file, `h5f` is HDF5 formatted input of file of your corpus (genrated like training set with `tools/mkiodata.py` like in `scripts/mktrain.sh`), `models` is a (list of) model file(s) to make perplexity evaluation.

## The other files' discription

### `modules/`

Foundamental models needed for the construction of transformer.

### `loss/`

Implementation of label smoothing loss function required by the training of transformer.

### `lrsch.py`

Learning rate schedule model needed according to the paper.

### `utils/`

Functions for basic features, for example, freeze / unfreeze parameters of models, padding list of tensors to same size on assigned dimension.

### `translator.py`

Provide an encapsulation for the whole translation procedure with which you can use the trained model in your application easier.

### `server.py`

An example depends on Flask to provide simple Web service and REST API about how to use the `translator`, configure [those variables](server.py#L13-L23) before you use it.

### `transformer/`

Implementations of seq2seq models.

### `parallel/`

Multi-GPU parallelization implementation.

### `datautils/`

Supportive functions for data segmentation.

### `tools/`

Scripts to support data processing (e.g. text to tensor), analyzing, model file handling, etc.

## Performance

Settings: WMT 2014, English -> German, 32k joint BPE with 8 as vocabulary threshold for BPE. 2 nVidia GTX 1080 Ti GPU(s) for training, 1 for decoding.

Tokenized case-sensitive BLEU measured with [multi-bleu.perl](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/generic/multi-bleu.perl), Training speed and decoding speed are measured by the number of target tokens (`<eos>` counted and `<pad>` discounted) per second and the number of sentences per second:

| | BLEU | Training Speed | Decoding Speed |
| :------| ------: | ------: | ------: |
| Attention is all you need | 27.3 | | |
| Neutron | 28.07 | 23213.65 | 150.15 |

## Acknowledgments

Hongfei Xu enjoys a doctoral grant from [China Scholarship Council](https://www.csc.edu.cn/) ([2018]3101, 201807040056) while maintaining this project.

Details of this project can be found [here](https://arxiv.org/abs/1903.07402), and please cite it if you enjoy the implementation :)

```
@article{xu2019neutron,
  author = {Xu, Hongfei and Liu, Qiuhui},
  title = "{Neutron: An Implementation of the Transformer Translation Model and its Variants}",
  journal = {arXiv preprint arXiv:1903.07402},
  archivePrefix = "arXiv",
  eprinttype = {arxiv},
  eprint = {1903.07402},
  primaryClass = "cs.CL",
  keywords = {Computer Science - Computation and Language},
  year = 2019,
  month = "March",
  url = {https://arxiv.org/abs/1903.07402},
  pdf = {https://arxiv.org/pdf/1903.07402}
}
```
