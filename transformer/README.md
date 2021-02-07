# Transformer

## `NMT.py`

The transformer model encapsulates encoder and decoder. Set [these lines](https://github.com/anoidgit/transformer/blob/master/transformer/NMT.py#L10-L14) to make a choice between the standard encoder / decoder and the others.

## `Encoder.py`

The encoder of transformer.

## `Decoder.py`

The standard decoder of transformer.

## `AvgDecoder.py`

The average decoder of transformer proposed by [Accelerating Neural Transformer via an Average Attention Network](https://www.aclweb.org/anthology/P18-1166/).

## `EnsembleNMT.py`

A model encapsulates several NMT models to do ensemble decoding. Configure [these lines](https://github.com/anoidgit/transformer/blob/master/transformer/EnsembleNMT.py#L8-L12) to make a choice between the standard decoder and the average decoder.

## `EnsembleEncoder.py`

A model encapsulates several encoders for ensemble decoding.

## `EnsembleDecoder.py`

A model encapsulates several standard decoders for ensemble decoding.

## `EnsembleAvgDecoder.py`

A model encapsulates several average decoders proposed by [Accelerating Neural Transformer via an Average Attention Network](https://www.aclweb.org/anthology/P18-1166/) for ensemble decoding.

## `AGG/`

Implementation of aggregation models.

### `Hier*.py`

Hierarchical aggregation proposed in [Exploiting Deep Representations for Neural Machine Translation](https://www.aclweb.org/anthology/D18-1457/).

## `TA/`

Implementation of transparent attention proposed in [Training Deeper Neural Machine Translation Models with Transparent Attention](https://aclweb.org/anthology/D18-1338/).

## `SC/`

Implementation of sentential context proposed in [Exploiting Sentential Context for Neural Machine Translation](https://www.aclweb.org/anthology/P19-1624/).

## `RealFormer/`

Implementation of [RealFormer: Transformer Likes Residual Attention](https://arxiv.org/abs/2012.11747) for machine translation.

## `Doc/`

Implementation of context-aware Transformer proposed in [Improving the Transformer Translation Model with Document-Level Context](https://www.aclweb.org/anthology/D18-1049/).

## `APE/`

Implementation of an APE model.
