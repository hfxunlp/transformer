#!/bin/bash

set -e -o pipefail -x

export cachedir=cache

export dataid=w19ape

export srcd=w19ape
export srcvf=dev/dev.src.tc
export mtvf=dev/dev.mt.tc
export tgtvf=dev/dev.pe.tc

export maxtokens=256

export bpeops=32000
export minfreq=8
export share_bpe=false

export tgtd=$cachedir/$dataid

# options for cleaning the data processed by bpe,
# advised values except numrules can be calculated by:
#	python tools/check/charatio.py $tgtd/src.dev.bpe $tgtd/tgt.dev.bpe, and
#	python tools/check/biratio.py $tgtd/src.dev.bpe $tgtd/tgt.dev.bpe
# with development set.
# As for numrules, choose from [1, 6], fewer data will be droped with larger value, none data would be droped if it was set to 6, details are described in:
#	tools/check/chars.py
export charatio=0.493
export bperatio=2.401
export seperatio=0.391
export bibperatio=2.251
export bioratio=2.501
export numrules=1

# cleaning bpe results and bpe again
python tools/clean/ape/chars.py $tgtd/src.train.bpe $tgtd/mt.train.bpe $tgtd/tgt.train.bpe $tgtd/src.clean.tmp $tgtd/mt.clean.tmp $tgtd/tgt.clean.tmp $charatio $bperatio $seperatio $bibperatio $bioratio $numrules

sed -r 's/(@@ )|(@@ ?$)//g' < $tgtd/src.clean.tmp > $tgtd/src.train.tok.clean
sed -r 's/(@@ )|(@@ ?$)//g' < $tgtd/mt.clean.tmp > $tgtd/mt.train.tok.clean
sed -r 's/(@@ )|(@@ ?$)//g' < $tgtd/tgt.clean.tmp > $tgtd/tgt.train.tok.clean
rm -fr $tgtd/src.clean.tmp $tgtd/mt.clean.tmp $tgtd/tgt.clean.tmp

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
subword-nmt apply-bpe -c $tgt_cdsf --vocabulary $tgtd/tgt.vcb.bpe --vocabulary-threshold $minfreq < $tgtd/mt.train.tok.clean > $tgtd/mt.train.bpe
subword-nmt apply-bpe -c $tgt_cdsf --vocabulary $tgtd/tgt.vcb.bpe --vocabulary-threshold $minfreq < $tgtd/tgt.train.tok.clean > $tgtd/tgt.train.bpe

subword-nmt apply-bpe -c $src_cdsf --vocabulary $tgtd/src.vcb.bpe --vocabulary-threshold $minfreq < $srcd/$srcvf > $tgtd/src.dev.bpe
subword-nmt apply-bpe -c $tgt_cdsf --vocabulary $tgtd/tgt.vcb.bpe --vocabulary-threshold $minfreq < $srcd/$mtvf > $tgtd/mt.dev.bpe
subword-nmt apply-bpe -c $tgt_cdsf --vocabulary $tgtd/tgt.vcb.bpe --vocabulary-threshold $minfreq < $srcd/$tgtvf > $tgtd/tgt.dev.bpe

# then execute scripts/mktrain.sh to generate training and development data.
