#!/bin/bash

set -e -o pipefail -x

# take the processed data from scripts/mkbpe.sh and convert to tensor representation.

export cachedir=cache
export dataid=opus

export srcd=$cachedir/$dataid
export srctf=src.train.bpe
export tgttf=tgt.train.bpe
export srcvf=src.dev.bpe
export tgtvf=tgt.dev.bpe

export rsf_train=train.h5
export rsf_dev=dev.h5

export share_vcb=true
export vsize=65536

export maxtokens=257

export ngpu=1

export do_sort=true
export build_vocab=true

export wkd=$cachedir/$dataid

mkdir -p $wkd

if $do_sort; then
	python tools/mulang/eff/sort.py $srcd/$srctf $srcd/$tgttf $wkd/src.train.srt $wkd/tgt.train.srt $maxtokens
	python tools/mulang/eff/sort.py $srcd/$srcvf $srcd/$tgtvf $wkd/src.dev.srt $wkd/tgt.dev.srt 1048576
fi

if $share_vcb; then
	export src_vcb=$wkd/common.vcb
	export tgt_vcb=$src_vcb
	if $build_vocab; then
		python tools/mulang/share_vocab.py $wkd/src.train.srt --target $wkd/tgt.train.srt $src_vcb $wkd/lang.vcb $vsize
		python tools/check/mulang/fbindexes.py $tgt_vcb $wkd/src.train.srt $wkd/tgt.train.srt $wkd/src.dev.srt $wkd/tgt.dev.srt $wkd/lang.vcb $wkd/fbind.py
	fi
else
	export src_vcb=$wkd/src.vcb
	export tgt_vcb=$wkd/tgt.vcb
	if $build_vocab; then
		python tools/mulang/vocab.py $wkd/src.train.srt $src_vcb $wkd/lang.vcb $vsize
		python tools/vocab.py $wkd/tgt.train.srt $tgt_vcb $vsize
		python tools/check/mulang/fbindexes.py $tgt_vcb $wkd/src.train.srt $wkd/tgt.train.srt $wkd/src.dev.srt $wkd/tgt.dev.srt $wkd/lang.vcb $wkd/fbind.py
	fi
fi

python tools/mulang/eff/mkiodata.py $wkd/src.train.srt $wkd/tgt.train.srt $src_vcb $tgt_vcb $wkd/lang.vcb $wkd/$rsf_train $ngpu
python tools/mulang/eff/mkiodata.py $wkd/src.dev.srt $wkd/tgt.dev.srt $src_vcb $tgt_vcb $wkd/lang.vcb $wkd/$rsf_dev $ngpu
