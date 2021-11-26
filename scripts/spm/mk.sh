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
export rratio=0.6
export maxtokens=256

export bpeops=32000
export minfreq=8
# 0.9995
export charcov=1.0
# unigram, bpe, char, or word
export mtype="unigram"
export share_bpe=true

export tgtd=$cachedir/$dataid

mkdir -p $tgtd

# clean the data first by removing different translations with lower frequency of same sentences
python tools/clean/maxkeeper.py $srcd/$srctf $srcd/$tgttf $tgtd/src.clean.tmp $tgtd/tgt.clean.tmp $maxtokens
python tools/clean/token_repeat.py $tgtd/src.clean.tmp $tgtd/tgt.clean.tmp $tgtd/src.clean.rtmp $tgtd/tgt.clean.rtmp $rratio
mv $tgtd/src.clean.rtmp $tgtd/src.clean.tmp
mv $tgtd/tgt.clean.rtmp $tgtd/tgt.clean.tmp

python tools/vocab.py $tgtd/src.clean.tmp $tgtd/src.full.vcb 1048576 &
python tools/vocab.py $tgtd/tgt.clean.tmp $tgtd/tgt.full.vcb 1048576 &
wait
python tools/clean/vocab/ratio.py $tgtd/src.clean.tmp $tgtd/tgt.clean.tmp $tgtd/src.train.tok.clean $tgtd/tgt.train.tok.clean $tgtd/src.full.vcb $tgtd/tgt.full.vcb $vratio
rm -fr $tgtd/src.full.vcb $tgtd/tgt.full.vcb $tgtd/src.clean.tmp $tgtd/tgt.clean.tmp

if $share_bpe; then
# to learn joint bpe
	export src_cdsf=$tgtd/bpe
	export tgt_cdsf=$tgtd/bpe
	cat $tgtd/src.train.tok.clean $tgtd/tgt.train.tok.clean | shuf > $tgtd/bpe.train.txt
	spm_train --input=$tgtd/bpe.train.txt --model_prefix=$src_cdsf --vocab_size=$bpeops --character_coverage=$charcov --model_type=$mtype --minloglevel=1
	spm_encode --model=$src_cdsf.model --generate_vocabulary < $tgtd/src.train.tok.clean > $tgtd/src.vcb.bpe &
	spm_encode --model=$tgt_cdsf.model --generate_vocabulary < $tgtd/tgt.train.tok.clean > $tgtd/tgt.vcb.bpe &
	wait
	rm $tgtd/bpe.train.txt
else
# to learn independent bpe:
	export src_cdsf=$tgtd/src
	export tgt_cdsf=$tgtd/tgt
	spm_train --input=$tgtd/src.train.tok.clean --model_prefix=$src_cdsf --vocab_size=$bpeops --character_coverage=$charcov --model_type=$mtype --minloglevel=1 &
	spm_train --input=$tgtd/tgt.train.tok.clean --model_prefix=$tgt_cdsf --vocab_size=$bpeops --character_coverage=$charcov --model_type=$mtype --minloglevel=1 &
	wait
	spm_encode --model=$src_cdsf.model --generate_vocabulary < $tgtd/src.train.tok.clean > $tgtd/src.vcb.bpe &
	spm_encode --model=$tgt_cdsf.model --generate_vocabulary < $tgtd/tgt.train.tok.clean > $tgtd/tgt.vcb.bpe &
	wait
fi

spm_encode --model=$src_cdsf.model --vocabulary=$tgtd/src.vcb.bpe --vocabulary_threshold=$minfreq < $tgtd/src.train.tok.clean > $tgtd/src.train.bpe &
spm_encode --model=$tgt_cdsf.model --vocabulary=$tgtd/tgt.vcb.bpe --vocabulary_threshold=$minfreq < $tgtd/tgt.train.tok.clean > $tgtd/tgt.train.bpe &

spm_encode --model=$src_cdsf.model --vocabulary=$tgtd/src.vcb.bpe --vocabulary_threshold=$minfreq < $srcd/$srcvf > $tgtd/src.dev.bpe &
spm_encode --model=$tgt_cdsf.model --vocabulary=$tgtd/tgt.vcb.bpe --vocabulary_threshold=$minfreq < $srcd/$tgtvf > $tgtd/tgt.dev.bpe &
wait

# report devlopment set features for cleaning
python tools/check/charatio.py $tgtd/src.dev.bpe $tgtd/tgt.dev.bpe
python tools/check/biratio.py $tgtd/src.dev.bpe $tgtd/tgt.dev.bpe
