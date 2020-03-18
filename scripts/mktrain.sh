#!/bin/bash

# take the processed data from scripts/mkbpe.sh and convert to tensor representation.

export cachedir=cache
export dataid=w14ed32

export srctf=src.train.bpe
export tgttf=tgt.train.bpe
export srcvf=src.dev.bpe
export tgtvf=tgt.dev.bpe

export share_vcb=false
export vsize=65536
export maxtokens=256
export ngpu=1

export wkd=$cachedir/$dataid

python tools/sort.py $wkd/$srctf $wkd/$tgttf $wkd/src.train.srt $wkd/tgt.train.srt $maxtokens
# use the following command to sort a very large dataset with limited memory
#bash tools/lsort/sort.sh $wkd/$srctf $wkd/$tgttf $wkd/src.train.srt $wkd/tgt.train.srt $maxtokens
python tools/sort.py $wkd/$srcvf $wkd/$tgtvf $wkd/src.dev.srt $wkd/tgt.dev.srt 1048576

if $share_vcb; then
	export src_vcb=$wkd/common.vcb
	export tgt_vcb=$src_vcb
	python tools/share_vocab.py $wkd/src.train.srt $wkd/tgt.train.srt $src_vcb $vsize
	python tools/check/fbindexes.py $src_vcb $wkd/tgt.train.srt $wkd/fbind.py
else
	export src_vcb=$wkd/src.vcb
	export tgt_vcb=$wkd/tgt.vcb
	python tools/vocab.py $wkd/src.train.srt $src_vcb $vsize
	python tools/vocab.py $wkd/tgt.train.srt $tgt_vcb $vsize
fi

python tools/mkiodata.py $wkd/src.train.srt $wkd/tgt.train.srt $src_vcb $tgt_vcb $wkd/train.h5 $ngpu
python tools/mkiodata.py $wkd/src.dev.srt $wkd/tgt.dev.srt $src_vcb $tgt_vcb $wkd/dev.h5 $ngpu
