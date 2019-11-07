#!/bin/bash

export srcd=w14ende
export srctf=test.tc.en.w14ed32
export modelf="expm/w14ende/checkpoint.t7"
export rsf=w14trs/trans.txt
export share_vcb=true

export cachedir=cache
export dataid=w14ed32

export ngpu=1

export tgtd=$cachedir/$dataid

export bpef=out.bpe

if [[ $share_vcb == true ]];
then
	export src_vcb=$tgtd/common.vcb
	export tgt_vcb=$src_vcb
else
	export src_vcb=$tgtd/src.vcb
	export tgt_vcb=$tgtd/tgt.vcb
fi

python tools/sorti.py $srcd/$srctf $tgtd/$srctf.srt
python tools/mktest.py $tgtd/$srctf.srt $src_vcb $tgtd/test.h5 $ngpu
python predict.py $tgtd/$bpef.srt $tgt_vcb $modelf
python tools/restore.py $srcd/$srctf $tgtd/$srctf.srt $tgtd/$bpef.srt $tgtd/$bpef
sed -r 's/(@@ )|(@@ ?$)//g' < $tgtd/$bpef > $rsf
