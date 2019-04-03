#!/bin/bash

# take the processed data from scripts/mkbpe.sh and convert to tensor representation.

export cachedir=cache
export dataid=un

export wkd=$cachedir/$dataid

export vsize=65536
export maxtokens=256
export ngpu=1

export srctf=src.train.bpe
export tgttf=tgt.train.bpe
export srcvf=src.dev.bpe
export tgtvf=tgt.dev.bpe

python tools/sort.py $wkd/$srctf $wkd/$tgttf $wkd/src.train.srt $wkd/tgt.train.srt $maxtokens
# use the following command to sort a very large dataset with limited memory
#bash tools/lsort/sort.sh $wkd/$srctf $wkd/$tgttf $wkd/src.train.srt $wkd/tgt.train.srt $maxtokens
python tools/sort.py $wkd/$srcvf $wkd/$tgtvf $wkd/src.dev.srt $wkd/tgt.dev.srt 1048576

python tools/vocab.py $wkd/src.train.srt $wkd/src.vcb $vsize
python tools/vocab.py $wkd/tgt.train.srt $wkd/tgt.vcb $vsize
# use the following line if you want a shared vocabulary
#python tools/share_vocab.py $wkd/src.train.srt $wkd/tgt.train.srt $wkd/common.vcb $vsize

python tools/mkiodata.py $wkd/src.train.srt $wkd/tgt.train.srt $wkd/src.vcb $wkd/tgt.vcb $wkd/train.h5 $ngpu
python tools/mkiodata.py $wkd/src.dev.srt $wkd/tgt.dev.srt $wkd/src.vcb $wkd/tgt.vcb $wkd/dev.h5 $ngpu
# use the following two lines if you want to share the embedding between encoder and decoder
#python tools/mkiodata.py $wkd/src.train.srt $wkd/tgt.train.srt $wkd/common.vcb $wkd/common.vcb $wkd/train.h5 $ngpu
#python tools/mkiodata.py $wkd/src.dev.srt $wkd/tgt.dev.srt $wkd/common.vcb $wkd/common.vcb $wkd/dev.h5 $ngpu

