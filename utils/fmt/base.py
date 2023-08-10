#encoding: utf-8

import sys
from bz2 import open as bz_open
from gzip import open as gz_open
from lzma import open as xz_open
from random import shuffle

from cnfg.hyp import raw_cache_compression_level
from cnfg.vocab.base import pad_id

serial_func, deserial_func = repr, eval

iter_to_str = lambda lin: map(str, lin)
iter_to_int = lambda lin: map(int, lin)
iter_to_float = lambda lin: map(float, lin)

def sys_open(fname, mode="r", compresslevel=raw_cache_compression_level, **kwargs):

	if fname == "-":
		return ((sys.stdin.buffer if "r" in mode else sys.stdout.buffer) if "b" in mode else (sys.stdin if "r" in mode else sys.stdout))
	else:
		if fname.endswith(".gz"):
			return gz_open(fname, mode=mode, compresslevel=compresslevel, **kwargs)
		elif fname.endswith(".bz2"):
			return bz_open(fname, mode=mode, compresslevel=compresslevel, **kwargs)
		elif fname.endswith(".xz"):
			return xz_open(fname, mode=mode, **kwargs)
		else:
			return open(fname, mode=mode, **kwargs)

def save_objects(fname, *inputs):

	ens = "\n".encode("utf-8")
	with sys_open(fname, "wb") as f:
		for tmpu in inputs:
			f.write(serial_func(tmpu).encode("utf-8"))
			f.write(ens)

def load_objects(fname):

	rs = []
	with sys_open(fname, "rb") as f:
		for line in f:
			tmp = line.strip()
			if tmp:
				rs.append(deserial_func(tmp.decode("utf-8")))

	return tuple(rs) if len(rs) > 1 else rs[0]

def load_states(fname):

	rs = []
	with sys_open(fname, "rb") as f:
		for line in f:
			tmp = line.strip()
			if tmp:
				for tmpu in tmp.decode("utf-8").split():
					if tmpu:
						rs.append(tmpu)

	return rs

def list_reader(fname, keep_empty_line=True, sep=None, print_func=print):

	with sys_open(fname, "rb") as frd:
		for line in frd:
			tmp = line.strip()
			if tmp:
				tmp = clean_list(tmp.decode("utf-8").split(sep=sep))
				yield tmp
			else:
				if print_func is not None:
					print_func("Reminder: encounter an empty line, which may not be the case.")
				if keep_empty_line:
					yield []

def line_reader(fname, keep_empty_line=True, print_func=print):

	with sys_open(fname, "rb") as frd:
		for line in frd:
			tmp = line.strip()
			if tmp:
				yield tmp.decode("utf-8")
			else:
				if print_func is not None:
					print_func("Reminder: encounter an empty line, which may not be the case.")
				if keep_empty_line:
					yield ""

def line_char_reader(fname, keep_empty_line=True, print_func=print):

	with sys_open(fname, "rb") as frd:
		for line in frd:
			tmp = line.strip()
			if tmp:
				yield list(tmp.decode("utf-8"))
			else:
				if print_func is not None:
					print_func("Reminder: encounter an empty line, which may not be the case.")
				if keep_empty_line:
					yield []

def list_reader_wst(fname, keep_empty_line=True, sep=None, print_func=print):

	with sys_open(fname, "rb") as frd:
		for line in frd:
			tmp = line.strip(b"\r\n")
			if tmp:
				tmp = clean_list(tmp.decode("utf-8").split(sep=sep))
				yield tmp
			else:
				if print_func is not None:
					print_func("Reminder: encounter an empty line, which may not be the case.")
				if keep_empty_line:
					yield []

def line_reader_wst(fname, keep_empty_line=True, print_func=print):

	with sys_open(fname, "rb") as frd:
		for line in frd:
			tmp = line.strip(b"\r\n")
			if tmp:
				yield tmp.decode("utf-8")
			else:
				if print_func is not None:
					print_func("Reminder: encounter an empty line, which may not be the case.")
				if keep_empty_line:
					yield ""

def loop_file_so(fsrc, frs, process_func=None, processor=None):

	ens = "\n".encode("utf-8")
	with sys_open(fsrc, "rb") as frd, sys_open(frs, "wb") as fwrt:
		for line in frd:
			tmp = line.strip()
			if tmp:
				fwrt.write(process_func(tmp.decode("utf-8"), processor).encode("utf-8"))
			fwrt.write(ens)

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

def maxfreq_filter_many(inputs):

	tmp = {}
	for _ in inputs:
		us, ut = tuple(_[:-1]), _[-1]
		if us in tmp:
			tmp[us][ut] = tmp[us].get(ut, 0) + 1
		else:
			tmp[us] = {ut: 1}

	rs = []
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
			rs.append((*tus, tut,))

	return rs

def maxfreq_filter_bi(inputs):

	tmp = {}
	for us, ut in inputs:
		if us in tmp:
			tmp[us][ut] = tmp[us].get(ut, 0) + 1
		else:
			tmp[us] = {ut: 1}

	rs = []
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
			rs.append((tus, tut,))

	return rs

def maxfreq_filter(inputs):

	return maxfreq_filter_many(inputs) if len(inputs[0]) > 2 else maxfreq_filter_bi(inputs)

def maxfreq_filter_core_pair(ls, lt):

	tmp = {}
	for us, ut in zip(ls, lt):
		if us in tmp:
			tmp[us][ut] = tmp[us].get(ut, 0) + 1
		else:
			tmp[us] = {ut: 1}

	rls, rlt = [], []
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

	return rls, rlt

def maxfreq_filter_pair(*inputs):

	if len(inputs) > 2:
		# here we assume that we only have one target and it is at the last position
		rsh, rst = maxfreq_filter_core_pair(tuple(zip(*inputs[:-1])), inputs[-1])
		return *zip(*rsh), rst
	else:
		return maxfreq_filter_core_pair(*inputs)

def shuffle_pair(*inputs):

	tmp = list(zip(*inputs))
	shuffle(tmp)

	return zip(*tmp)

def get_bsize(maxlen, maxtoken, maxbsize):

	rs = max(maxtoken // maxlen, 1)
	if (rs % 2 == 1) and (rs > 1):
		rs -= 1

	return min(rs, maxbsize)

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

def iter_dict_sort(dict_in, reverse=False, free=False):

	d_keys = list(dict_in.keys())
	d_keys.sort(reverse=reverse)
	for d_key in d_keys:
		d_v = dict_in[d_key]
		if isinstance(d_v, dict):
			yield from iter_dict_sort(d_v, reverse=reverse, free=free)
		else:
			yield d_v
	if free:
		dict_in.clear()

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

def seperate_list_iter(lin, k):

	i = 0
	_ = []
	for lu in lin:
		_.append(lu)
		i += 1
		if i >= k:
			yield _
			_ = []
			i = 0
	if _:
		yield _

def seperate_list(lin, k):

	return list(seperate_list_iter(lin, k))

def all_in(lin, setin):

	return all(lu in setin for lu in lin)

def all_le(lin, value):

	return all(lu <= value for lu in lin)

def all_gt(lin, value):

	return all(lu > value for lu in lin)

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

def pad_batch(i_d, mlen_i, pad_id=pad_id):

	if isinstance(i_d[0], (tuple, list,)):
		return [pad_batch(idu, mlen_i, pad_id=pad_id) for idu in i_d]
	else:
		curlen = len(i_d)
		if curlen < mlen_i:
			i_d.extend([pad_id for i in range(mlen_i - curlen)])
		return i_d

class FileList(list):

	def __init__(self, files, *inputs, **kwargs):

		super(FileList, self).__init__(sys_open(fname, *inputs, **kwargs) for fname in files)

	def __enter__(self):

		return self

	def __exit__(self, *inputs, **kwargs):

		for _f in self:
			_f.close()

def multi_line_reader(fname, *inputs, num_line=1, **kwargs):

	_i = 0
	rs = []
	ens = "\n".encode("utf-8") if ("rb" in inputs) or ("rb" in kwargs.values()) else "\n"
	with sys_open(fname, *inputs, **kwargs) as frd:
		for line in frd:
			tmp = line.rstrip()
			rs.append(tmp)
			_i += 1
			if _i >= num_line:
				yield ens.join(rs)
				rs = []
				_i = 0
	if rs:
		yield ens.join(rs)

def read_lines(fin, num_lines):

	_last_ind = num_lines - 1
	for i, _ in enumerate(fin, 1):
		yield _
		if i > _last_ind:
			break
