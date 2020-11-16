#!/bin/bash

set -e -o pipefail -x

# take the processed data from scripts/mkbpe.sh and convert to tensor representation.

export cachedir=cache
export dataid=w14ed32

export srcd=$cachedir/$dataid
export srctf=src.train.bpe
export tgttf=tgt.train.bpe
export srcvf=src.dev.bpe
export tgtvf=tgt.dev.bpe

export rsf_train=train.h5
export rsf_dev=dev.h5

export share_vcb=false
export vsize=65536

export maxtokens=256

export ngpu=1

export wkd=$cachedir/$dataid

mkdir -p $wkd

python tools/sort.py $srcd/$srctf $srcd/$tgttf $wkd/src.train.srt $wkd/tgt.train.srt $maxtokens
# use the following command to sort a very large dataset with limited memory
#bash tools/lsort/sort.sh $srcd/$srctf $srcd/$tgttf $wkd/src.train.srt $wkd/tgt.train.srt $maxtokens
python tools/sort.py $srcd/$srcvf $srcd/$tgtvf $wkd/src.dev.srt $wkd/tgt.dev.srt 1048576

if $share_vcb; then
	export src_vcb=$wkd/common.vcb
	export tgt_vcb=$src_vcb
	python tools/share_vocab.py $wkd/src.train.srt $wkd/tgt.train.srt $src_vcb $vsize
	python tools/check/fbindexes.py $tgt_vcb $wkd/tgt.train.srt $wkd/tgt.dev.srt $wkd/fbind.py
else
	export src_vcb=$wkd/src.vcb
	export tgt_vcb=$wkd/tgt.vcb
	python tools/vocab.py $wkd/src.train.srt $src_vcb $vsize
	python tools/vocab.py $wkd/tgt.train.srt $tgt_vcb $vsize
fi

python tools/mkiodata.py $wkd/src.train.srt $wkd/tgt.train.srt $src_vcb $tgt_vcb $wkd/$rsf_train $ngpu
python tools/mkiodata.py $wkd/src.dev.srt $wkd/tgt.dev.srt $src_vcb $tgt_vcb $wkd/$rsf_dev $ngpu
