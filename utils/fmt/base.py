#encoding: utf-8

from random import shuffle

from cnfg.hyp import use_unk

pad_id, sos_id, eos_id = 0, 1, 2
if use_unk:
	unk_id = 3
	init_vocab = {"<pad>":pad_id, "<sos>":sos_id, "<eos>":eos_id, "<unk>":unk_id}
	init_normal_token_id = 4
else:
	unk_id = None
	init_vocab = {"<pad>":pad_id, "<sos>":sos_id, "<eos>":eos_id}
	init_normal_token_id = 3
init_token_id = 3

def tostr(lin):

	return [str(lu) for lu in lin]

def save_states(fname, stl):

	with open(fname, "wb") as f:
		f.write(" ".join(stl).encode("utf-8"))
		f.write("\n".encode("utf-8"))

def load_states(fname):

	rs = []
	with open(fname, "rb") as f:
		for line in f:
			tmp = line.strip()
			if tmp:
				for tmpu in tmp.decode("utf-8").split():
					if tmpu:
						rs.append(tmpu)

	return rs

def list_reader(fname):

	with open(fname, "rb") as frd:
		for line in frd:
			tmp = line.strip()
			if tmp:
				tmp = clean_list(tmp.decode("utf-8").split())
				yield tmp

def line_reader(fname):

	with open(fname, "rb") as frd:
		for line in frd:
			tmp = line.strip()
			if tmp:
				yield tmp.decode("utf-8")

def ldvocab(vfile, minf=False, omit_vsize=False, vanilla=False):

	global init_vocab, init_normal_token_id

	if vanilla:
		rs, cwd = {}, 0
	else:
		rs, cwd = init_vocab.copy(), init_normal_token_id
	if omit_vsize:
		vsize = omit_vsize
	else:
		vsize = False
	for data in list_reader(vfile):
		freq = int(data[0])
		if (not minf) or freq > minf:
			if vsize:
				ndata = len(data) - 1
				if vsize >= ndata:
					for wd in data[1:]:
						rs[wd] = cwd
						cwd += 1
				else:
					for wd in data[1:vsize + 1]:
						rs[wd] = cwd
						cwd += 1
						ndata = vsize
					break
				vsize -= ndata
				if vsize <= 0:
					break
			else:
				for wd in data[1:]:
					rs[wd] = cwd
					cwd += 1
		else:
			break
	return rs, cwd

def save_vocab(vcb_dict, fname, omit_vsize=False):

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
	with open(fname, "wb") as f:
		for freq in freqs:
			cdata = r_vocab[freq]
			ndata = len(cdata) - 1
			if remain and (remain < ndata):
				cdata = cdata[:remain + 1]
				ndata = remain
			f.write(" ".join(cdata).encode("utf-8"))
			f.write(ens)
			if remain:
				remain -= ndata
				if remain <= 0:
					break

def reverse_dict(din):

	return {v:k for k, v in din.items()}

def ldvocab_list(vfile, minf=False, omit_vsize=False):

	rs = []
	if omit_vsize:
		vsize = omit_vsize
	else:
		vsize = False
	cwd = 0
	for data in list_reader(vfile):
		freq = int(data[0])
		if (not minf) or freq > minf:
			if vsize:
				ndata = len(data) - 1
				if vsize >= ndata:
					rs.extend(data[1:])
					cwd += ndata
				else:
					rs.extend(data[1:vsize + 1])
					cwd += vsize
					break
				vsize -= ndata
				if vsize <= 0:
					break
			else:
				rs.extend(data[1:])
				cwd += len(data) - 1
		else:
			break

	return rs, cwd

def clean_str(strin):

	return " ".join([tmpu for tmpu in strin.split() if tmpu])

def clean_list(lin):

	return [tmpu for tmpu in lin if tmpu]

def clean_list_iter(lin):

	for lu in lin:
		if lu:
			yield lu

def clean_liststr_lentok(lin):

	rs = [tmpu for tmpu in lin if tmpu]

	return " ".join(rs), len(rs)

def maxfreq_filter(ls, lt, max_remove=True):

	tmp = {}
	for us, ut in zip(ls, lt):
		if us not in tmp:
			tmp[us] = {ut: 1} if max_remove else set([ut])
		else:
			if max_remove:
				tmp[us][ut] = tmp[us].get(ut, 0) + 1
			elif ut not in tmp[us]:
				tmp[us].add(ut)
	rls, rlt = [], []
	if max_remove:
		for tus, tlt in tmp.items():
			_rs = []
			_maxf = 0
			for key, value in tlt.items():
				if value > _maxf:
					_maxf = value
					_rs = [key]
				elif value == _maxf:
					_rs.append(key)
			for tut in _rs:
				rls.append(tus)
				rlt.append(tut)
	else:
		for tus, tlt in tmp.items():
			for tut in tlt:
				rls.append(tus)
				rlt.append(tut)

	return rls, rlt

def shuffle_pair(*inputs):

	tmp = list(zip(*inputs))
	shuffle(tmp)

	return zip(*tmp)

def get_bsize(maxlen, maxtoken, maxbsize):

	rs = max(maxtoken // maxlen, 1)
	if (rs % 2 == 1) and (rs > 1):
		rs -= 1

	return min(rs, maxbsize)

def no_unk_mapper(vcb, ltm, prompt=False):

	if prompt:
		rs = []
		for wd in ltm:
			if wd in vcb:
				rs.append(vcb[wd])
			else:
				print("Error mapping: "+ wd)
		return rs
	else:
		return [vcb[wd] for wd in ltm if wd in vcb]

def list2dict(lin, kfunc=None):

	return {k: lu for k, lu in enumerate(lin)} if kfunc is None else {kfunc(k): lu for k, lu in enumerate(lin)}

def dict_is_list(sdin, kfunc=None):

	_lset = set(range(len(sdin))) if kfunc is None else set(kfunc(i) for i in range(len(sdin)))

	return False if (_lset - sdin) else True

def dict2pairs(dict_in):

	rsk = []
	rsv = []
	for key, value in dict_in.items():
		rsk.append(key)
		rsv.append(value)

	return rsk, rsv

def iter_dict_sort(dict_in, reverse=False):

	d_keys = list(dict_in.keys())
	d_keys.sort(reverse=reverse)
	for d_key in d_keys:
		d_v = dict_in[d_key]
		if isinstance(d_v, dict):
			for _item in iter_dict_sort(d_v, reverse):
				yield _item
		else:
			yield d_v

def dict_insert_set(dict_in, value, *keys):

	if len(keys) > 1:
		_cur_key = keys[0]
		dict_in[_cur_key] = dict_insert_set(dict_in.get(_cur_key, {}), value, *keys[1:])
	else:
		key = keys[0]
		if key in dict_in:
			_dset = dict_in[key]
			if value not in _dset:
				_dset.add(value)
		else:
			dict_in[key] = set([value])

	return dict_in

def dict_insert_list(dict_in, value, *keys):

	if len(keys) > 1:
		_cur_key = keys[0]
		dict_in[_cur_key] = dict_insert_list(dict_in.get(_cur_key, {}), value, *keys[1:])
	else:
		key = keys[0]
		if key in dict_in:
			dict_in[key].append(value)
		else:
			dict_in[key] = [value]

	return dict_in

def legal_vocab(sent, ilgset, ratio):

	total = ilg = 0
	for tmpu in sent.split():
		if tmpu:
			if tmpu in ilgset:
				ilg += 1
			total += 1
	rt = float(ilg) / float(total)

	return False if rt > ratio else True

def all_in(lin, sin):

	for lu in lin:
		if not lu in sin:
			return False

	return True

def get_char_ratio(strin):

	ntokens = nchars = nsp = 0
	pbpe = False
	for tmpu in strin.split():
		if tmpu:
			if tmpu.endswith("@@"):
				nchars += 1
				if not pbpe:
					pbpe = True
					nsp += 1
			elif pbpe:
				pbpe = False
			ntokens += 1
	lorigin = float(len(strin.replace("@@ ", "").split()))
	ntokens = float(ntokens)

	return float(nchars) / ntokens, ntokens / lorigin, float(nsp) / lorigin

def get_bi_ratio(ls, lt):

	if ls > lt:
		return float(ls) / float(lt)
	else:
		return float(lt) / float(ls)

def map_batch(i_d, vocabi):

	global use_unk, sos_id, eos_id, unk_id

	if isinstance(i_d[0], (tuple, list,)):
		return [map_batch(idu, vocabi)[0] for idu in i_d], 2
	else:
		rsi = [sos_id]
		rsi.extend([vocabi.get(wd, unk_id) for wd in i_d] if use_unk else no_unk_mapper(vocabi, i_d))#[vocabi[wd] for wd in i_d if wd in vocabi]
		rsi.append(eos_id)
		return rsi, 2

def pad_batch(i_d, mlen_i):

	global pad_id

	if isinstance(i_d[0], (tuple, list,)):
		return [pad_batch(idu, mlen_i) for idu in i_d]
	else:
		curlen = len(i_d)
		if curlen < mlen_i:
			i_d.extend([pad_id for i in range(mlen_i - curlen)])
		return i_d

def parse_none(vin, value):

	return value if vin is None else vin

def parse_double_value_tuple(vin):

	if isinstance(vin, (list, tuple,)):
		return vin[0], vin[-1]
	else:
		return vin, vin
