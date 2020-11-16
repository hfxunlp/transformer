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

export debpe=true

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

python tools/sort.py $srcd/$srctf $srcd/$srcmf $tgtd/$srctf.srt $tgtd/$srcmf.srt 1048576
python tools/mkiodata.py $tgtd/$srctf.srt $tgtd/$srcmf.srt $src_vcb $tgt_vcb $tgtd/test.h5 $ngpu
python predict_ape.py $tgtd/$bpef.srt $tgt_vcb $modelf
python tools/ape/restore.py $srcd/$srctf $srcd/$srcmf $tgtd/$srctf.srt $tgtd/$srcmf.srt $tgtd/$bpef.srt $tgtd/$bpef
if $debpe; then
	sed -r 's/(@@ )|(@@ ?$)//g' < $tgtd/$bpef > $rsf
	rm $tgtd/$bpef
else
	mv $tgtd/$bpef $rsf
fi
rm $tgtd/$srctf.srt $tgtd/$srcmf.srt $tgtd/$bpef.srt
