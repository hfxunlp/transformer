#!/bin/bash

set -e -o pipefail -x

# take the processed data from scripts/bpe/mk|clean.sh and convert to tensor representation.

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

export do_sort=true
export build_vocab=true

export faext=".xz"

export wkd=$cachedir/$dataid

mkdir -p $wkd

export stsf=$wkd/src.train.srt$faext
export mtsf=$wkd/mt.train.srt$faext
export ttsf=$wkd/tgt.train.srt$faext
export sdsf=$wkd/src.dev.srt$faext
export mdsf=$wkd/mt.dev.srt$faext
export tdsf=$wkd/tgt.dev.srt$faext
if $do_sort; then
	python tools/sort.py $srcd/$srctf $srcd/$mttf $srcd/$tgttf $stsf $mtsf $ttsf $maxtokens &
	python tools/sort.py $srcd/$srcvf $srcd/$mtvf $srcd/$tgtvf $sdsf $mdsf $tdsf 1048576 &
	wait
fi

if $share_vcb; then
	export src_vcb=$wkd/common.vcb
	export tgt_vcb=$src_vcb
	if $build_vocab; then
		python tools/vocab/token/share.py $stsf $ttsf $mtsf $src_vcb $vsize
		python tools/check/fbindexes.py $tgt_vcb $ttsf $tdsf $wkd/fbind.py &
	fi
else
	export src_vcb=$wkd/src.vcb
	export tgt_vcb=$wkd/tgt.vcb
	if $build_vocab; then
		python tools/vocab/token/single.py $stsf $src_vcb $vsize &
		python tools/vocab/token/share.py $ttsf $mtsf $tgt_vcb $vsize &
		wait
	fi
fi

python tools/ape/mkiodata.py $stsf $mtsf $ttsf $src_vcb $tgt_vcb $wkd/$rsf_train $ngpu &
python tools/ape/mkiodata.py $sdsf $mdsf $tdsf $src_vcb $tgt_vcb $wkd/$rsf_dev $ngpu &
wait
