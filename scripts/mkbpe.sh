#!/bin/bash

export cachedir=cache
export srcd=un-data

export dataid=un

export bpeops=32000
export minfreq=50
export maxtokens=256


export srctf=tok.zh
export tgttf=tok.en
export srcvf=06.tok.zh
export tgtvf=06.tok.en0

export tgtd=$cachedir/$dataid

mkdir -p $tgtd

# clean the data first by removing different translations with lower frequency of same sentences
python tools/clean/maxkeeper.py $srcd/$srctf $srcd/$tgttf $tgtd/src.train.tok.clean $tgtd/tgt.train.tok.clean $maxtokens

# to learn joint bpe
#subword-nmt learn-joint-bpe-and-vocab --input  $tgtd/src.train.tok.clean $tgtd/tgt.train.tok.clean -s $bpeops -o $tgtd/bpe.cds --write-vocabulary $tgtd/src.vcb.bpe $tgtd/tgt.vcb.bpe
# to learn independent bpe:
subword-nmt learn-bpe -s $bpeops < $tgtd/src.train.tok.clean > $tgtd/src.cds
subword-nmt learn-bpe -s $bpeops < $tgtd/tgt.train.tok.clean > $tgtd/tgt.cds
subword-nmt get-vocab --input $tgtd/src.train.tok.clean --output $tgtd/src.vcb.bpe
subword-nmt get-vocab --input $tgtd/tgt.train.tok.clean --output $tgtd/tgt.vcb.bpe

# to apply joint bpe for train set:
#subword-nmt apply-bpe -c $tgtd/bpe.cds --vocabulary $tgtd/src.vcb.bpe --vocabulary-threshold $minfreq < $tgtd/src.train.tok.clean > $tgtd/src.train.bpe
#subword-nmt apply-bpe -c $tgtd/bpe.cds --vocabulary $tgtd/tgt.vcb.bpe --vocabulary-threshold $minfreq < $tgtd/tgt.train.tok.clean > $tgtd/tgt.train.bpe
# to apply independent bpe for train set:
subword-nmt apply-bpe -c $tgtd/src.cds --vocabulary $tgtd/src.vcb.bpe --vocabulary-threshold $minfreq < $tgtd/src.train.tok.clean > $tgtd/src.train.bpe
subword-nmt apply-bpe -c $tgtd/tgt.cds --vocabulary $tgtd/tgt.vcb.bpe --vocabulary-threshold $minfreq < $tgtd/tgt.train.tok.clean > $tgtd/tgt.train.bpe
# to apply joint bpe for development set:
#subword-nmt apply-bpe -c $tgtd/bpe.cds --vocabulary $tgtd/src.vcb.bpe --vocabulary-threshold $minfreq < $srcd/$srcvf > $tgtd/src.dev.bpe
#subword-nmt apply-bpe -c $tgtd/bpe.cds --vocabulary $tgtd/tgt.vcb.bpe --vocabulary-threshold $minfreq < $srcd/$tgtvf > $tgtd/tgt.dev.bpe
# to apply independent bpe for development set:
subword-nmt apply-bpe -c $tgtd/src.cds --vocabulary $tgtd/src.vcb.bpe --vocabulary-threshold $minfreq < $srcd/$srcvf > $tgtd/src.dev.bpe
subword-nmt apply-bpe -c $tgtd/tgt.cds --vocabulary $tgtd/tgt.vcb.bpe --vocabulary-threshold $minfreq < $srcd/$tgtvf > $tgtd/tgt.dev.bpe

# report devlopment set features for cleaning
python tools/check/charatio.py $tgtd/src.dev.bpe $tgtd/tgt.dev.bpe
python tools/check/biratio.py $tgtd/src.dev.bpe $tgtd/tgt.dev.bpe
