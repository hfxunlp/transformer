#!/bin/bash

export srcd=wmt14
export srctf=test.tc.en.w14ed32
export modelf="expm/w14ed32/std/base/checkpoint.h5"
export rsd=w14trs
export rsf=$rsd/trans.txt

export share_vcb=false

export cachedir=cache
export dataid=w14ed32

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

python tools/sorti.py $srcd/$srctf $tgtd/$srctf.srt
python tools/mktest.py $tgtd/$srctf.srt $src_vcb $tgtd/test.h5 $ngpu
python predict.py $tgtd/$bpef.srt $tgt_vcb $modelf
python tools/restore.py $srcd/$srctf $tgtd/$srctf.srt $tgtd/$bpef.srt $tgtd/$bpef
if $debpe; then
	sed -r 's/(@@ )|(@@ ?$)//g' < $tgtd/$bpef > $rsf
	rm $tgtd/$bpef
else
	mv $tgtd/$bpef $rsf
fi
rm $tgtd/$srctf.srt $tgtd/$bpef.srt
