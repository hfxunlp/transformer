#encoding: utf-8

import sys

from utils.fmt.base import clean_list, iter_to_str, loop_file_so
from utils.fmt.vocab.base import map_instance, no_unk_mapper
from utils.fmt.vocab.token import ldvocab

from cnfg.vocab.base import eos_id, init_normal_token_id, init_vocab, sos_id, unk_id, use_unk

def handle(srcf, vcbf, rsf, add_sp_tokens=True, minfreq=False, vsize=False):

	_vcb = ldvocab(vcbf, minf=minfreq, omit_vsize=vsize, vanilla=False, init_vocab=init_vocab, init_normal_token_id=init_normal_token_id)[0]
	map_line = (lambda lin, vcb: " ".join(iter_to_str(map_instance(clean_list(lin.split()), vcb, use_unk=use_unk, sos_id=sos_id, eos_id=eos_id, unk_id=unk_id)))) if add_sp_tokens else ((lambda lin, vcb: " ".join(iter_to_str(vcb.get(wd, unk_id) for wd in clean_list(lin.split())))) if use_unk else (lambda lin, vcb: " ".join(iter_to_str(no_unk_mapper(vcb, clean_list(lin.split()))))))

	return loop_file_so(srcf, rsf, process_func=map_line, processor=_vcb)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3])
