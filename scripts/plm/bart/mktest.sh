#!/bin/bash

set -e -o pipefail -x

export srcd=~/cache/bart
export srctf=src.test.txt
export modelf=""
export rsd=$srcd
export rsf=$rsd/dec.test.txt
export src_vcb=~/plm/bart-base/tokenizer.json

export cachedir=cache
export dataid=bart

export ngpu=1

export sort_decode=true

export tgtd=$cachedir/$dataid

export tgt_vcb=$src_vcb
export bpef=out.bpe

mkdir -p $rsd

python tools/plm/map/bart.py $srcd/$srctf $src_vcb $tgtd/$srctf.ids
if $sort_decode; then
	export srt_input_f=$tgtd/$srctf.ids.srt
	python tools/sort.py $tgtd/$srctf.ids $srt_input_f 1048576
else
	export srt_input_f=$tgtd/$srctf.ids
fi

python tools/plm/mktest.py $srt_input_f $tgtd/test.h5 $ngpu
python predict_bart.py $tgtd/$bpef $tgt_vcb $modelf

if $sort_decode; then
	python tools/restore.py $tgtd/$srctf.ids $srt_input_f $tgtd/$bpef $rsf
	rm $srt_input_f $tgtd/$bpef
else
	mv $tgtd/$bpef $rsf
fi
