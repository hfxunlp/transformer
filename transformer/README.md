# Transformer

## `NMT.py`

The transformer model encapsulates encoder and decoder. Set [these lines](NMT.py#L6-L9) to make a choice between the standard encoder / decoder and the others.

## `Encoder.py`

The encoder of transformer.

## `Decoder.py`

The standard decoder of transformer.

## `AvgDecoder.py`

The average decoder of transformer proposed by [Accelerating Neural Transformer via an Average Attention Network](https://aclanthology.org/P18-1166/).

## `EnsembleNMT.py`

A model encapsulates several NMT models to do ensemble decoding. Configure [these lines](EnsembleNMT.py#L6-L9) to make a choice between the standard decoder and the average decoder.

## `EnsembleEncoder.py`

A model encapsulates several encoders for ensemble decoding.

## `EnsembleDecoder.py`

A model encapsulates several standard decoders for ensemble decoding.

## `EnsembleAvgDecoder.py`

A model encapsulates several average decoders proposed by [Accelerating Neural Transformer via an Average Attention Network](https://aclanthology.org/P18-1166/) for ensemble decoding.

## `AGG/`

Implementation of aggregation models.

### `Hier*.py`

Hierarchical aggregation proposed in [Exploiting Deep Representations for Neural Machine Translation](https://aclanthology.org/D18-1457/).

## `TA/`

Implementation of transparent attention proposed in [Training Deeper Neural Machine Translation Models with Transparent Attention](https://aclanthology.org/D18-1338/).

## `SC/`

Implementation of sentential context proposed in [Exploiting Sentential Context for Neural Machine Translation](https://aclanthology.org/P19-1624/).

## `SDU/`

Implementation of self-dependency units proposed in [Highway Transformer: Self-Gating Enhanced Self-Attentive Networks](https://aclanthology.org/2020.acl-main.616/).

## `RealFormer/`

Implementation of [RealFormer: Transformer Likes Residual Attention](https://arxiv.org/abs/2012.11747) for machine translation.

## `LD/`

Implementation of NMT with phrase representations proposed in [Learning Source Phrase Representations for Neural Machine Translation](https://aclanthology.org/2020.acl-main.37/).

## `Doc/`

Implementation of context-aware Transformer proposed in [Improving the Transformer Translation Model with Document-Level Context](https://aclanthology.org/D18-1049/).

## `APE/`

Implementation of an APE model.

## `PLM/`

Implementation of pre-trained language models.

## `Prompt/`

Implementation of PLM-based prompt learning models.
