#encoding: utf-8

import sys

from utils.fmt.base import list_reader as file_reader, sys_open

from cnfg.vocab.base import init_normal_token_id, init_vocab

sep_load, sep_save = None, " "

def ldvocab(vfile, minf=False, omit_vsize=False, vanilla=False, init_vocab=init_vocab, init_normal_token_id=init_normal_token_id, sep=sep_load, file_reader=file_reader, print_func=print):

	if vanilla:
		rs, cwd = {}, 0
	else:
		rs, cwd = init_vocab.copy(), len(init_vocab) if init_normal_token_id is None else init_normal_token_id
	if omit_vsize:
		vsize = max(0, omit_vsize - cwd)
	else:
		vsize = False
	dkeys = set()
	for data in file_reader(vfile, keep_empty_line=False, sep=sep):
		freq = int(data[0])
		if (not minf) or (freq > minf):
			if vsize:
				ndata = len(data) - 1
				if vsize > ndata:
					for wd in data[1:]:
						if wd in rs:
							if wd not in dkeys:
								dkeys.add(wd)
							ndata -= 1
						else:
							rs[wd] = cwd
							cwd += 1
					vsize -= ndata
				else:
					_break = False
					for wd in data[1:]:
						if wd in rs:
							if wd not in dkeys:
								dkeys.add(wd)
						else:
							rs[wd] = cwd
							cwd += 1
							vsize -= 1
							_break = (vsize <= 0)
							if _break:
								break
					if _break:
						break
			else:
				for wd in data[1:]:
					if wd in rs:
						if wd not in dkeys:
							dkeys.add(wd)
					else:
						rs[wd] = cwd
						cwd += 1
		else:
			break
	if (print_func is not None) and dkeys:
		print_func("duplicated vocab keys: %s" % str(dkeys))

	return rs, cwd

def save_vocab(vcb_dict, fname, omit_vsize=False, sep=sep_save):

	r_vocab = {}
	for k, v in vcb_dict.items():
		if v not in r_vocab:
			r_vocab[v]=[str(v), k]
		else:
			r_vocab[v].append(k)

	freqs = list(r_vocab.keys())
	freqs.sort(reverse=True)

	ens = "\n".encode("utf-8")
	remain = omit_vsize
	with sys_open(fname, "wb") as f:
		for freq in freqs:
			cdata = r_vocab[freq]
			ndata = len(cdata) - 1
			if remain and (remain < ndata):
				cdata = cdata[:remain + 1]
				ndata = remain
			f.write(sep.join(cdata).encode("utf-8"))
			f.write(ens)
			if remain:
				remain -= ndata
				if remain <= 0:
					break

def ldvocab_list(vfile, minf=False, omit_vsize=False, sep=sep_load, file_reader=file_reader, print_func=print):

	rs = []
	if omit_vsize:
		vsize = omit_vsize
	else:
		vsize = False
	cwd, lwd, dkeys = 0, set(), set()
	for data in file_reader(vfile, keep_empty_line=False, sep=sep):
		freq = int(data[0])
		if (not minf) or (freq > minf):
			if vsize:
				ndata = len(data) - 1
				if vsize > ndata:
					for wd in data[1:]:
						if wd in lwd:
							if wd not in dkeys:
								dkeys.add(wd)
							ndata -= 1
						else:
							rs.append(wd)
							lwd.add(wd)
							cwd += 1
					vsize -= ndata
				else:
					_break = False
					for wd in data[1:]:
						if wd in lwd:
							if wd not in dkeys:
								dkeys.add(wd)
						else:
							rs.append(wd)
							lwd.add(wd)
							cwd += 1
							vsize -= 1
							_break = (vsize <= 0)
							if _break:
								break
					if _break:
						break
			else:
				for wd in data[1:]:
					if wd in lwd:
						if wd not in dkeys:
							dkeys.add(wd)
					else:
						rs.append(wd)
						lwd.add(wd)
						cwd += 1
		else:
			break
	if (print_func is not None) and dkeys:
		print_func("duplicated vocab keys: %s" % str(dkeys))

	return rs, cwd

def ldvocab_freq(vfile, minf=False, omit_vsize=False, sep=sep_load, file_reader=file_reader, print_func=print):

	rs = {}
	if omit_vsize:
		vsize = omit_vsize
	else:
		vsize = False
	cwd = 0
	dkeys = set()
	for data in file_reader(vfile, keep_empty_line=False, sep=sep):
		freq = int(data[0])
		if (not minf) or (freq > minf):
			if vsize:
				ndata = len(data) - 1
				if vsize > ndata:
					for wd in data[1:]:
						if wd in rs:
							if wd not in dkeys:
								dkeys.add(wd)
							ndata -= 1
						else:
							rs[wd] = freq
					cwd += ndata
					vsize -= ndata
				else:
					_break = False
					for wd in data[1:]:
						if wd in rs:
							if wd not in dkeys:
								dkeys.add(wd)
						else:
							rs[wd] = freq
							cwd += 1
							vsize -= 1
							_break = (vsize <= 0)
							if _break:
								break
					if _break:
						break
			else:
				for wd in data[1:]:
					if wd in rs:
						if wd not in dkeys:
							dkeys.add(wd)
					else:
						rs[wd] = freq
						cwd += 1
		else:
			break
	if (print_func is not None) and dkeys:
		print_func("duplicated vocab keys: %s" % str(dkeys))

	return rs, cwd
