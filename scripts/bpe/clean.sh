#!/bin/bash

export cachedir=cache
export srcd=w14ende

export dataid=w14ed32

export bpeops=32000
export minfreq=8
export maxtokens=256

export srcvf=dev.tc.en
export tgtvf=dev.tc.de

# options for cleaning the data processed by bpe,
# advised values except numrules can be calculated by:
#	python tools/check/charatio.py $tgtd/src.dev.bpe $tgtd/tgt.dev.bpe, and
#	python tools/check/biratio.py $tgtd/src.dev.bpe $tgtd/tgt.dev.bpe
# with development set.
# As for numrules, choose from [1, 6], fewer data will be droped with larger value, none data would be droped if it was set to 6, details are described in:
#	tools/check/chars.py
export charatio=0.973
export bperatio=36.01
export seperatio=1.01
export bibperatio=7.51
export bioratio=7.51
export numrules=1

export tgtd=$cachedir/$dataid

# cleaning bpe results and bpe again
python tools/clean/chars.py $tgtd/src.train.bpe $tgtd/tgt.train.bpe $tgtd/src.clean.tmp $tgtd/tgt.clean.tmp $charatio $bperatio $seperatio $bibperatio $bioratio $numrules

sed -r 's/(@@ )|(@@ ?$)//g' < $tgtd/src.clean.tmp > $tgtd/src.train.tok.clean
sed -r 's/(@@ )|(@@ ?$)//g' < $tgtd/tgt.clean.tmp > $tgtd/tgt.train.tok.clean
rm -fr $tgtd/src.clean.tmp $tgtd/tgt.clean.tmp

# to learn joint bpe
subword-nmt learn-joint-bpe-and-vocab --input $tgtd/src.train.tok.clean $tgtd/tgt.train.tok.clean -s $bpeops -o $tgtd/bpe.cds --write-vocabulary $tgtd/src.vcb.bpe $tgtd/tgt.vcb.bpe
# to learn independent bpe:
#subword-nmt learn-bpe -s $bpeops < $tgtd/src.train.tok.clean > $tgtd/src.cds
#subword-nmt learn-bpe -s $bpeops < $tgtd/tgt.train.tok.clean > $tgtd/tgt.cds
#subword-nmt apply-bpe -c $tgtd/src.cds < $tgtd/src.train.tok.clean | subword-nmt get-vocab > $tgtd/src.vcb.bpe
#subword-nmt apply-bpe -c $tgtd/tgt.cds < $tgtd/tgt.train.tok.clean | subword-nmt get-vocab > $tgtd/tgt.vcb.bpe

# to apply joint bpe for train set:
subword-nmt apply-bpe -c $tgtd/bpe.cds --vocabulary $tgtd/src.vcb.bpe --vocabulary-threshold $minfreq < $tgtd/src.train.tok.clean > $tgtd/src.train.bpe
subword-nmt apply-bpe -c $tgtd/bpe.cds --vocabulary $tgtd/tgt.vcb.bpe --vocabulary-threshold $minfreq < $tgtd/tgt.train.tok.clean > $tgtd/tgt.train.bpe
# to apply independent bpe for train set:
#subword-nmt apply-bpe -c $tgtd/src.cds --vocabulary $tgtd/src.vcb.bpe --vocabulary-threshold $minfreq < $tgtd/src.train.tok.clean > $tgtd/src.train.bpe
#subword-nmt apply-bpe -c $tgtd/tgt.cds --vocabulary $tgtd/tgt.vcb.bpe --vocabulary-threshold $minfreq < $tgtd/tgt.train.tok.clean > $tgtd/tgt.train.bpe

# to apply joint bpe for development set:
subword-nmt apply-bpe -c $tgtd/bpe.cds --vocabulary $tgtd/src.vcb.bpe --vocabulary-threshold $minfreq < $srcd/$srcvf > $tgtd/src.dev.bpe
subword-nmt apply-bpe -c $tgtd/bpe.cds --vocabulary $tgtd/tgt.vcb.bpe --vocabulary-threshold $minfreq < $srcd/$tgtvf > $tgtd/tgt.dev.bpe
# to apply independent bpe for development set:
#subword-nmt apply-bpe -c $tgtd/src.cds --vocabulary $tgtd/src.vcb.bpe --vocabulary-threshold $minfreq < $srcd/$srcvf > $tgtd/src.dev.bpe
#subword-nmt apply-bpe -c $tgtd/tgt.cds --vocabulary $tgtd/tgt.vcb.bpe --vocabulary-threshold $minfreq < $srcd/$tgtvf > $tgtd/tgt.dev.bpe

# then execute scripts/mktrain.sh to generate training and development data.
