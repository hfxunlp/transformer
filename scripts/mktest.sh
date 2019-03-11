#!/bin/bash

export srcd=un-cache
export srctf=src-val.bpe
export modelf="expm/debug/checkpoint.t7"
export rsf=trans.txt
export ngpu=1

export cachedir=cache
export dataid=un

export tgtd=$cachedir/$dataid

export bpef=out.bpe

python tools/sorti.py $srcd/$srctf $tgtd/$srctf.srt
python tools/mktest.py $tgtd/$srctf.srt $tgtd/src.vcb $tgtd/test.h5 $ngpu
python predict.py $tgtd/$bpef.srt $tgtd/tgt.vcb $modelf
python tools/restore.py $srcd/$srctf $tgtd/$srctf.srt $tgtd/$bpef.srt $tgtd/$bpef
sed -r 's/(@@ )|(@@ ?$)//g' < $tgtd/$bpef > $rsf
