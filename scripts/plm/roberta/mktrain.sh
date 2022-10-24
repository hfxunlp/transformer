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

if $do_sort; then
	python tools/sort.py $srcd/$srctf $srcd/$tgttf $wkd/src.train.srt $wkd/tgt.train.srt $maxtokens &
	# use the following command to sort a very large dataset with limited memory
	#bash tools/lsort/sort.sh $srcd/$srctf $srcd/$tgttf $wkd/src.train.srt $wkd/tgt.train.srt $maxtokens &
	python tools/sort.py $srcd/$srcvf $srcd/$tgtvf $wkd/src.dev.srt $wkd/tgt.dev.srt 1048576 &
	wait
fi

python tools/plm/mkiodata.py $wkd/src.train.srt $wkd/tgt.train.srt $wkd/$rsf_train $ngpu &
python tools/plm/mkiodata.py $wkd/src.dev.srt $wkd/tgt.dev.srt $wkd/$rsf_dev $ngpu &
wait
