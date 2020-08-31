#!/bin/bash

# take the processed data from scripts/mkbpe.sh and convert to tensor representation.

export cachedir=cache
export dataid=w19ape

export srcd=$cachedir/$dataid
export srctf=src.train.bpe
export mttf=mt.train.bpe
export tgttf=tgt.train.bpe
export srcvf=src.dev.bpe
export mtvf=tgt.dev.bpe
export tgtvf=tgt.dev.bpe

export rsf_train=train.h5
export rsf_dev=dev.h5

export share_vcb=false
export vsize=65536

export maxtokens=256

export ngpu=1

export wkd=$cachedir/$dataid

python tools/ape/sort.py $srcd/$srctf $srcd/$mttf $srcd/$tgttf $wkd/src.train.srt $wkd/mt.train.srt $wkd/tgt.train.srt $maxtokens
python tools/ape/sort.py $srcd/$srcvf $srcd/$mtvf $srcd/$tgtvf $wkd/src.dev.srt $wkd/mt.dev.srt $wkd/tgt.dev.srt 1048576

if $share_vcb; then
	export src_vcb=$wkd/common.vcb
	export tgt_vcb=$src_vcb
	python tools/share_vocab.py $wkd/src.train.srt $wkd/tgt.train.srt $wkd/mt.train.srt $src_vcb $vsize
	python tools/check/fbindexes.py $tgt_vcb $wkd/tgt.train.srt $wkd/tgt.dev.srt $wkd/fbind.py
else
	export src_vcb=$wkd/src.vcb
	export tgt_vcb=$wkd/tgt.vcb
	python tools/vocab.py $wkd/src.train.srt $src_vcb $vsize
	python tools/share_vocab.py $wkd/tgt.train.srt $wkd/mt.train.srt $tgt_vcb $vsize
fi

python tools/ape/mkiodata.py $wkd/src.train.srt $wkd/mt.train.srt $wkd/tgt.train.srt $src_vcb $tgt_vcb $wkd/$rsf_train $ngpu
python tools/ape/mkiodata.py $wkd/src.dev.srt $wkd/mt.dev.srt $wkd/tgt.dev.srt $src_vcb $tgt_vcb $wkd/$rsf_dev $ngpu
