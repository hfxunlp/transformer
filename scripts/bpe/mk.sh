#!/bin/bash

set -e -o pipefail -x

export cachedir=cache

export dataid=w14ed32

export srcd=w14ende
export srctf=train.tc.en
export tgttf=train.tc.de
export srcvf=dev.tc.en
export tgtvf=dev.tc.de

export vratio=0.2
export maxtokens=256

export bpeops=32000
export minfreq=8
export share_bpe=false

export tgtd=$cachedir/$dataid

mkdir -p $tgtd

# clean the data first by removing different translations with lower frequency of same sentences
python tools/clean/maxkeeper.py $srcd/$srctf $srcd/$tgttf $tgtd/src.clean.tmp $tgtd/tgt.clean.tmp $maxtokens

python tools/vocab.py $tgtd/src.clean.tmp $tgtd/src.full.vcb 1048576
python tools/vocab.py $tgtd/tgt.clean.tmp $tgtd/tgt.full.vcb 1048576
python tools/clean/vocab/ratio.py $tgtd/src.clean.tmp $tgtd/tgt.clean.tmp $tgtd/src.train.tok.clean $tgtd/tgt.train.tok.clean $tgtd/src.full.vcb $tgtd/tgt.full.vcb $vratio
rm -fr $tgtd/src.full.vcb $tgtd/tgt.full.vcb $tgtd/src.clean.tmp $tgtd/tgt.clean.tmp

if $share_bpe; then
# to learn joint bpe
	export src_cdsf=$tgtd/bpe.cds
	export tgt_cdsf=$tgtd/bpe.cds
	subword-nmt learn-joint-bpe-and-vocab --input $tgtd/src.train.tok.clean $tgtd/tgt.train.tok.clean -s $bpeops -o $src_cdsf --write-vocabulary $tgtd/src.vcb.bpe $tgtd/tgt.vcb.bpe
else
# to learn independent bpe:
	export src_cdsf=$tgtd/src.cds
	export tgt_cdsf=$tgtd/tgt.cds
	subword-nmt learn-bpe -s $bpeops < $tgtd/src.train.tok.clean > $src_cdsf
	subword-nmt learn-bpe -s $bpeops < $tgtd/tgt.train.tok.clean > $tgt_cdsf
	subword-nmt apply-bpe -c $src_cdsf < $tgtd/src.train.tok.clean | subword-nmt get-vocab > $tgtd/src.vcb.bpe
	subword-nmt apply-bpe -c $tgt_cdsf < $tgtd/tgt.train.tok.clean | subword-nmt get-vocab > $tgtd/tgt.vcb.bpe
fi

subword-nmt apply-bpe -c $src_cdsf --vocabulary $tgtd/src.vcb.bpe --vocabulary-threshold $minfreq < $tgtd/src.train.tok.clean > $tgtd/src.train.bpe
subword-nmt apply-bpe -c $tgt_cdsf --vocabulary $tgtd/tgt.vcb.bpe --vocabulary-threshold $minfreq < $tgtd/tgt.train.tok.clean > $tgtd/tgt.train.bpe

subword-nmt apply-bpe -c $src_cdsf --vocabulary $tgtd/src.vcb.bpe --vocabulary-threshold $minfreq < $srcd/$srcvf > $tgtd/src.dev.bpe
subword-nmt apply-bpe -c $tgt_cdsf --vocabulary $tgtd/tgt.vcb.bpe --vocabulary-threshold $minfreq < $srcd/$tgtvf > $tgtd/tgt.dev.bpe

# report devlopment set features for cleaning
python tools/check/charatio.py $tgtd/src.dev.bpe $tgtd/tgt.dev.bpe
python tools/check/biratio.py $tgtd/src.dev.bpe $tgtd/tgt.dev.bpe
