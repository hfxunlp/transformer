#!/bin/bash

set -e -o pipefail -x

export srcd=w19ape/test
export srctf=test.src.tc.w19ape
export srcmf=test.mt.tc.w19ape
export modelf="expm/w19ape/std/base/avg.h5"
export rsd=w19apetrs/std
export rsf=$rsd/base_avg.txt

export share_vcb=false

export cachedir=cache
export dataid=w19ape

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
	export srt_input_fm=$tgtd/$srcmf.srt$faext
	python tools/sort.py $srcd/$srctf $srcd/$srcmf $srt_input_f $srt_input_fm 1048576
else
	export srt_input_f=$srcd/$srctf
	export srt_input_fm=$srcd/$srcmf
fi

python tools/mkiodata.py $srt_input_f $srt_input_fm $src_vcb $tgt_vcb $tgtd/test.h5 $ngpu
python predict_ape.py $tgtd/$bpef.srt $tgt_vcb $modelf

if $sort_decode; then
	python tools/restore.py $srcd/$srctf $srcd/$srcmf $srt_input_f $srt_input_fm $tgtd/$bpef.srt $tgtd/$bpef
	rm $srt_input_f $srt_input_fm $tgtd/$bpef.srt
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
