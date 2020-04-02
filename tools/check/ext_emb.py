#encoding: utf-8

''' usage:
	python tools/ext_emb.py vocab_file emb_file result
'''

import sys

import torch

from utils.fmt.base import ldvocab, reverse_dict
from utils.fmt.base4torch import load_emb_txt
from utils.h5serial import h5save, h5load

def handle(vcbf, embf, rsf):

	vcb, nwd = ldvocab(vcbf)
	emb = load_emb_txt(vcb, embf)
	unkemb = emb.get("<unk>", torch.zeros(emb[list(emb.keys())[0]].size(0)))
	vcb = reverse_dict(vcb)
	rs = []
	for i in range(nwd):
		rs.append(emb.get(vcb[i], unkemb))
	h5save(torch.stack(rs, 0), rsf)

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3])
