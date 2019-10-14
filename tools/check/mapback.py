#encoding: utf-8

import sys

import numpy, h5py
import torch

from utils.fmt.base import ldvocab, reverse_dict

def handle(h5f, vcbsf, vcbtf, rsfs, rsft):

	td = h5py.File(h5f, "r")

	ntest = td["ndata"][:].item()
	nword = td["nwordi"][:].tolist()
	nwordi = nword[0]
	vcbs, nwords = ldvocab(vcbsf)
	vcbs = reverse_dict(vcbs)
	vcbt, nwordt = ldvocab(vcbtf)
	vcbt = reverse_dict(vcbt)
	src_grp, tgt_grp = td["src"], td["tgt"]

	ens = "\n".encode("utf-8")

	with open(rsfs, "wb") as fs:
		with open(rsft, "wb") as ft:
			for i in range(ntest):
				curid = str(i)
				curd = torch.from_numpy(src_grp[curid][:]).tolist()
				md = []
				for iu in curd:
					md.append(" ".join([vcbs.get(i) for i in iu]))
				fs.write("\n".join(md).encode("utf-8"))
				fs.write(ens)
				curd = torch.from_numpy(tgt_grp[curid][:]).tolist()
				md = []
				for tu in curd:
					md.append(" ".join([vcbt.get(i) for i in tu]))
				ft.write("\n".join(md).encode("utf-8"))
				ft.write(ens)

	td.close()

if __name__ == "__main__":
	handle(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
