#!/bin/bash

set -e -o pipefail -x

export srcsf=$1
export srctf=$2
export rssf=$3
export rstf=$4
export maxtokens=$5

export cachedir=cache/lsort

rm -fr $cachedir
mkdir -p $cachedir

python tools/lsort/partsort.py $srcsf $srctf $cachedir $maxtokens
python tools/lsort/merge.py $cachedir $rssf $rstf
rm -fr $cachedir
