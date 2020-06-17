# datautils

## `bpe.py`

A tool borrowed from [subword-nmt](https://github.com/rsennrich/subword-nmt) to apply bpe for `translator`.

## `moses.py`

Codes to encapsulate moses scripts, you have to define `moses_scripts`(path to moses scripts) and ensure `perl` is executable to use it, otherwise, you need to modify [these two lines](https://github.com/anoidgit/transformer/blob/master/datautils/moses.py#L7-L8) to tell the module where to find them.

## `pymoses.py`

Wrapping of sacremoses implementation of moses scripts.

## `zh.py`

Chinese segmentation is different from tokenization, a tool is provided to support Chinese based on [pynlpir](https://github.com/tsroten/pynlpir).
