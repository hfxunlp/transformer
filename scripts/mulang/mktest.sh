#!/bin/bash

set -e -o pipefail -x

export srcd=opus
export srctf=$1
export modelf="expm/opus/std/base/checkpoint.h5"
export rsd=opurs
export rsf=$rsd/trans.txt

export share_vcb=true

export cachedir=cache
export dataid=opus

export ngpu=1

export sort_decode=true
export debpe=true
export spm_bpe=false

export faext=".xz"

export tgtd=$cachedir/$dataid

export bpef=out.bpe

if $share_vcb; then
	export src_vcb=$tgtd/common.vcb
	export tgt_vcb=$src_vcb
else
	export src_vcb=$tgtd/src.vcb
	export tgt_vcb=$tgtd/tgt.vcb
fi

mkdir -p $rsd

if $sort_decode; then
	export srt_input_f=$tgtd/$srctf.srt$faext
	python tools/mulang/eff/sort.py $srcd/$srctf $srt_input_f 1048576
else
	export srt_input_f=$srcd/$srctf
fi

python tools/mulang/eff/mktest.py $srt_input_f $src_vcb $tgtd/lang.vcb $tgtd/test.h5 $ngpu
python predict_mulang.py $tgtd/$bpef.srt $tgt_vcb $modelf

if $sort_decode; then
	python tools/restore.py $srcd/$srctf $srt_input_f $tgtd/$bpef.srt $tgtd/$bpef
	rm $srt_input_f $tgtd/$bpef.srt
else
	mv $tgtd/$bpef.srt $tgtd/$bpef
fi

if $debpe; then
	if $spm_bpe; then
		python tools/spm/decode.py --model $tgtd/bpe.model --input_format piece --input $tgtd/$bpef > $rsf

	else
		sed -r 's/(@@ )|(@@ ?$)//g' < $tgtd/$bpef > $rsf
	fi
	rm $tgtd/$bpef
else
	mv $tgtd/$bpef $rsf
fi
rm $tgtd/test.h5
