#!/bin/bash

set -e -o pipefail -x

export srcd=w19edoc
export srctf=test.en.w19edoc
export modelf="expm/w19edoc/doc/base/checkpoint.h5"
export rsd=w19edoctrs
export rsf=$rsd/trans.txt

export share_vcb=false

export cachedir=cache
export dataid=w19edoc

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

python tools/doc/mono/sort.py $srcd/$srctf $tgtd/$srctf.srt
python tools/doc/para/mktest.py $tgtd/$srctf.srt $src_vcb $tgtd/test.h5 $ngpu
python predict_doc_para.py $tgtd/$bpef.srt $tgt_vcb $modelf
python tools/doc/para/restore.py $srcd/$srctf w19ed/test.en.w19ed w19edtrs/base_avg.tbrs $tgtd/$srctf.srt $tgtd/$bpef.srt $tgtd/$bpef
if $debpe; then
	sed -r 's/(@@ )|(@@ ?$)//g' < $tgtd/$bpef > $rsf
	rm $tgtd/$bpef
else
	mv $tgtd/$bpef $rsf
fi
rm $tgtd/$srctf.srt $tgtd/$bpef.srt
