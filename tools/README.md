# tools

## `average_model.py`

A tool to average several models to one which may bring some additional performance with no additional costs. Example usage:

`python tools/average_model.py $averaged_model_file.h5 $model1.h5 $model2.h5 ...`

## `sort.py`

Sort the dataset to make the training more easier and start from easier questions.

## `vocab.py`

Build vocabulary for the training set.

## `mkiodata.py`

Convert text data to hdf5 format for the training script. Settings for the training data like batch size, maximum tokens per batch unit and padding limitation can be found [here](https://github.com/anoidgit/transformer/blob/master/cnfg/hyp.py#L20-L24).

## `mktest.py`

Convert translation requests to hdf5 format for the prediction script. Settings for the test data like batch size, maximum tokens per batch unit and padding limitation can be found [here](https://github.com/anoidgit/transformer/blob/master/cnfg/hyp.py#L20-L24).

## `lsort/`

Scripts to support sorting very large training set with limited memory.

## `check/`

### `debug/`
Tools to check the implementation and the data.

### `fbindexes.py`

When you using a shared vocabulary for source side and target side, there are still some words which only appear at the source side even joint BPE is applied. Those words take up probabilities in the label smoothing classifier, and this tool can prevent this through generating a larger and well covered forbidden indexes list which can be concatnated to `forbidden_indexes` in `cnfg/base.py`.

## `clean/`

Tools to filter the datasets.

