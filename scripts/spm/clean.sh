#!/bin/bash

set -e -o pipefail -x

export cachedir=cache

export dataid=w14ed32

export srcd=w14ende
export srcvf=dev.tc.en
export tgtvf=dev.tc.de

export maxtokens=256

export bpeops=32000
export minfreq=8
# 0.9995
export charcov=1.0
# unigram, bpe, char, or word
export mtype="unigram"
export share_bpe=true

export tgtd=$cachedir/$dataid

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

# cleaning bpe results and bpe again
python tools/clean/chars.py $tgtd/src.train.bpe $tgtd/tgt.train.bpe $tgtd/src.clean.tmp $tgtd/tgt.clean.tmp $charatio $bperatio $seperatio $bibperatio $bioratio $numrules

if $share_bpe; then
	export src_cdsf=$tgtd/bpe
	export tgt_cdsf=$tgtd/bpe
else
	export src_cdsf=$tgtd/src
	export tgt_cdsf=$tgtd/tgt
fi

spm_decode --model=$src_cdsf.model --input_format=piece < $tgtd/src.clean.tmp > $tgtd/src.train.tok.clean &
spm_decode --model=$tgt_cdsf.model --input_format=piece < $tgtd/tgt.clean.tmp > $tgtd/tgt.train.tok.clean &
wait
rm -fr $tgtd/src.clean.tmp $tgtd/tgt.clean.tmp

if $share_bpe; then
# to learn joint bpe
	# --max_sentence_length=4096 --input_sentence_size=5000000 --shuffle_input_sentence=true --num_threads=32 --train_extremely_large_corpus=true
	spm_train --input=$tgtd/src.train.tok.clean,$tgtd/tgt.train.tok.clean --model_prefix=$src_cdsf --vocab_size=$bpeops --character_coverage=$charcov --model_type=$mtype --unk_id=3 --bos_id=1 --eos_id=2 --pad_id=0 --unk_piece="<mask>" --bos_piece="<sos>" --eos_piece="<eos>" --unk_surface="<unk>" --minloglevel=1 --random_seed=666666
	spm_encode --model=$src_cdsf.model --generate_vocabulary < $tgtd/src.train.tok.clean > $tgtd/src.vcb.bpe &
	spm_encode --model=$tgt_cdsf.model --generate_vocabulary < $tgtd/tgt.train.tok.clean > $tgtd/tgt.vcb.bpe &
	wait
else
# to learn independent bpe:
	spm_train --input=$tgtd/src.train.tok.clean --model_prefix=$src_cdsf --vocab_size=$bpeops --character_coverage=$charcov --model_type=$mtype --unk_id=3 --bos_id=1 --eos_id=2 --pad_id=0 --unk_piece="<mask>" --bos_piece="<sos>" --eos_piece="<eos>" --unk_surface="<unk>" --minloglevel=1 --random_seed=666666 &
	spm_train --input=$tgtd/tgt.train.tok.clean --model_prefix=$tgt_cdsf --vocab_size=$bpeops --character_coverage=$charcov --model_type=$mtype --unk_id=3 --bos_id=1 --eos_id=2 --pad_id=0 --unk_piece="<mask>" --bos_piece="<sos>" --eos_piece="<eos>" --unk_surface="<unk>" --minloglevel=1 --random_seed=666666 &
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

# then execute scripts/mktrain.sh to generate training and development data.
