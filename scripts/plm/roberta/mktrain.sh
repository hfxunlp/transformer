#!/bin/bash

set -e -o pipefail -x

# take the processed data from scripts/bpe/mk|clean.sh and convert to tensor representation.

export cachedir=cache
export dataid=sst2

export srcd=$cachedir/$dataid
export srctf=src.train.ids
export tgttf=tgt.train.ids
export srcvf=src.dev.ids
export tgtvf=tgt.dev.ids

export rsf_train=train.h5
export rsf_dev=dev.h5

export maxtokens=512

export ngpu=1

export do_sort=true

export wkd=$cachedir/$dataid

mkdir -p $wkd

export stsf=$wkd/src.train.srt$faext
export ttsf=$wkd/tgt.train.srt$faext
export sdsf=$wkd/src.dev.srt$faext
export tdsf=$wkd/tgt.dev.srt$faext
if $do_sort; then
	python tools/sort.py $srcd/$srctf $srcd/$tgttf $stsf $ttsf $maxtokens &
	# use the following command to sort a very large dataset with limited memory
	#bash tools/lsort/sort.sh $srcd/$srctf $srcd/$tgttf $stsf $ttsf $maxtokens &
	python tools/sort.py $srcd/$srcvf $srcd/$tgtvf $sdsf $tdsf 1048576 &
	wait
fi

python tools/plm/mkiodata.py $stsf $ttsf $wkd/$rsf_train $ngpu &
python tools/plm/mkiodata.py $sdsf $tdsf $wkd/$rsf_dev $ngpu &
wait
