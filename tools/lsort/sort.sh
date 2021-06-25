#!/bin/bash

set -e -o pipefail -x

export nfiles=$[($# - 1) / 2]
export srcfl=${@: 1: nfiles}
export tgtfl=${@: $[nfiles + 1]: nfiles}
export maxtokens=${@: -1}

export cachedir=cache/lsort

rm -fr $cachedir
mkdir -p $cachedir

python tools/lsort/partsort.py $srcfl $cachedir $maxtokens
python tools/lsort/merge.py $cachedir $tgtfl
rm -fr $cachedir
