#encoding: utf-8

import torch

from transformer.NMT import NMT
from transformer.EnsembleNMT import NMT as Ensemble
from parallel.parallelMT import DataParallelMT

from utils import *

has_unk = True

def clear_list(lin):
	rs = []
	for tmpu in lin:
		if tmpu:
			rs.append(tmpu)
	return rs

def clean_len(line):
	rs = clear_list(line.split())
	return " ".join(rs), len(rs)

def clean_list(lin):
	rs = []
	for lu in lin:
		rs.append(" ".join(clear_list(lu.split())))
	return rs

def list_reader(fname):

	with open(fname, "rb") as frd:
		for line in frd:
			tmp = line.strip()
			if tmp:
				tmp = clear_list(tmp.decode("utf-8").split())
				yield tmp

def ldvocab(vfile, minf = False, omit_vsize = False):
	global has_unk
	if has_unk:
		rs = {"<pad>":0, "<sos>":1, "<eos>":2, "<unk>":3}
		cwd = 4
	else:
		rs = {"<pad>":0, "<sos>":1, "<eos>":2}
		cwd = 3
	vsize = False
	if omit_vsize:
		vsize = omit_vsize
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

def reverse_dict(din):
	rs = {}
	for k, v in din.items():
		rs[v] = k
	return rs

def batch_loader(sentences_iter, bsize, maxpad, maxpart, maxtoken, minbsize):
	def get_bsize(maxlen, maxtoken, maxbsize):
		rs = max(maxtoken // maxlen, 1)
		if (rs % 2 == 1) and (rs > 1):
			rs -= 1
		return min(rs, maxbsize)
	def clear_reader(lin):
		for line in lin:
			rs = []
			for tmpu in line.strip().split():
				if tmpu:
					rs.append(tmpu)
			yield rs
	rsi = []
	nd = 0
	maxlen = 0
	minlen = 0
	_bsize = bsize
	mlen_i = 0
	for i_d in clear_reader(sentences_iter):
		lgth = len(i_d)
		if maxlen == 0:
			_maxpad = max(1, min(maxpad, lgth // maxpart + 1) // 2)
			maxlen = lgth + _maxpad
			minlen = lgth - _maxpad
			_bsize = get_bsize(maxlen, maxtoken, bsize)
		if (nd < minbsize) or (lgth <= maxlen and lgth >= minlen and nd < _bsize):
			rsi.append(i_d)
			if lgth > mlen_i:
				mlen_i = lgth
			nd += 1
		else:
			yield rsi, mlen_i
			rsi = [i_d]
			mlen_i = lgth
			_maxpad = max(1, min(maxpad, lgth // maxpart + 1) // 2)
			maxlen = lgth + _maxpad
			minlen = lgth - _maxpad
			_bsize = get_bsize(maxlen, maxtoken, bsize)
			nd = 1
	if rsi:
		yield rsi, mlen_i

def batch_mapper(finput, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize):
	global has_unk
	for i_d, mlen_i in batch_loader(finput, bsize, maxpad, maxpart, maxtoken, minbsize):
		rsi = []
		for lined in i_d:
			tmp = [1]
			tmp.extend([vocabi.get(wd, 3) for wd in lined] if has_unk else [vocabi[wd] for wd in lined if wd in vocabi])
			tmp.append(2)
			rsi.append(tmp)
		yield rsi, mlen_i + 2

def batch_padder(finput, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize):
	for i_d, mlen_i in batch_mapper(finput, vocabi, bsize, maxpad, maxpart, maxtoken, minbsize):
		rid = []
		for lined in i_d:
			curlen = len(lined)
			if curlen < mlen_i:
				lined.extend([0 for i in range(mlen_i - curlen)])
			rid.append(lined)
		yield rid

def data_loader(sentences_iter, vcbi, minbsize=1, bsize=64, maxpad=16, maxpart=4, maxtoken=1536):
	for i_d in batch_padder(sentences_iter, vcbi, bsize, maxpad, maxpart, maxtoken, minbsize):
		yield torch.tensor(i_d, dtype=torch.long)

def load_fixing(module):
	if "fix_load" in dir(module):
		module.fix_load()

def sorti(lin):

	data = {}

	for ls in lin:
		ls = ls.strip()
		if ls:
			ls, lgth = clean_len(ls)
			if lgth not in data:
				data[lgth] = set([ls])
			elif ls not in data[lgth]:
				data[lgth].add(ls)

	length = list(data.keys())
	length.sort()

	rs = []

	for lgth in length:
		rs.extend(data[lgth])

	return rs

def restore(src, tsrc, trs):

	data = {}

	for sl, tl in zip(tsrc, trs):
		_sl, _tl = sl.strip(), tl.strip()
		if _sl and _tl:
			data[_sl] = " ".join(clear_list(_tl.split()))

	rs = []
	_tl = []
	for line in src:
		tmp = line.strip()
		if tmp:
			tmp = " ".join(clear_list(tmp.split()))
			tmp = data.get(tmp, "").strip()
			if tmp:
				_tl.append(tmp)
			elif _tl:
				rs.append(" ".join(_tl))
				_tl = []
		elif _tl:
			rs.append(" ".join(_tl))
			_tl = []
		else:
			rs.append("")
	if _tl:
		rs.append(" ".join(_tl))

	return rs

class TranslatorCore:

	def __init__(self, modelfs, fvocab_i, fvocab_t, cnfg, minbsize=1, expand_for_mulgpu=True, bsize=64, maxpad=16, maxpart=4, maxtoken=1536, minfreq = False, vsize = False):

		vcbi, nwordi = ldvocab(fvocab_i, minfreq, vsize)
		vcbt, nwordt = ldvocab(fvocab_t, minfreq, vsize)
		self.vcbi, self.vcbt = vcbi, reverse_dict(vcbt)

		if expand_for_mulgpu:
			self.bsize = bsize * minbsize
			self.maxtoken = maxtoken * minbsize
		else:
			self.bsize = bsize
			self.maxtoken = maxtoken
		self.maxpad = maxpad
		self.maxpart = maxpart
		self.minbsize = minbsize

		if isinstance(modelfs, (list, tuple)):
			models = []
			for modelf in modelfs:
				tmp = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.share_emb, cnfg.nhead, cnfg.cache_len, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes)

				tmp = load_model_cpu(modelf, tmp)
				tmp.apply(load_fixing)

				models.append(tmp)
			model = Ensemble(models)

		else:
			model = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.share_emb, cnfg.nhead, cnfg.cache_len, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes)

			model = load_model_cpu(modelfs, model)
			model.apply(load_fixing)
				
		cuda_device = torch.device(cnfg.gpuid)

		model.eval()

		use_cuda = cnfg.use_cuda
		gpuid = cnfg.gpuid
		if use_cuda and torch.cuda.is_available():
			use_cuda = True
			if len(gpuid.split(",")) > 1:
				if cnfg.multi_gpu_decoding:
					cuda_device = torch.device(gpuid[:gpuid.find(",")].strip())
					cuda_devices = [int(_.strip()) for _ in gpuid[gpuid.find(":") + 1:].split(",")]
					multi_gpu = True
				else:
					cuda_device = torch.device("cuda:" + gpuid[gpuid.rfind(","):].strip())
					multi_gpu = False
					cuda_devices = None
			else:
				cuda_device = torch.device(gpuid)
				multi_gpu = False
				cuda_devices = None
			torch.cuda.set_device(cuda_device.index)
		else:
			cuda_device = False
			multi_gpu = False
			cuda_devices = None
		if use_cuda:
			model.to(cuda_device)
			if multi_gpu:
				model = DataParallelMT(model, device_ids=cuda_devices, output_device=cuda_device.index, host_replicate=True, gather_output=False)
		self.use_cuda = use_cuda
		self.cuda_device = cuda_device
		self.beam_size = cnfg.beam_size
		self.multi_gpu = multi_gpu

		self.length_penalty = cnfg.length_penalty
		self.net = model

	def __call__(self, sentences_iter):
		rs = []
		with torch.no_grad():
			for seq_batch in data_loader(sentences_iter, self.vcbi, self.minbsize, self.bsize, self.maxpad, self.maxpart, self.maxtoken):
				if self.use_cuda:
					seq_batch = seq_batch.to(self.cuda_device)
				output = self.net.decode(seq_batch, self.beam_size, None, self.length_penalty)
				if self.multi_gpu:
					tmp = []
					for ou in output:
						tmp.extend(ou.tolist())
					output = tmp
				else:
					output = output.tolist()
				for tran in output:
					tmp = []
					for tmpu in tran:
						if (tmpu == 2) or (tmpu == 0):
							break
						else:
							tmp.append(self.vcbt[tmpu])
					rs.append(" ".join(tmp))
		return rs

class Translator:

	def __init__(self, trans=None, sent_split=None, tok=None, detok=None, bpe=None, debpe=None, punc_norm=None, truecaser=None, detruecaser=None):

		self.sent_split = sent_split

		self.flow = []
		if punc_norm is not None:
			self.flow.append(punc_norm)
		if tok is not None:
			self.flow.append(tok)
		if truecaser is not None:
			self.flow.append(truecaser)
		if bpe is not None:
			self.flow.append(bpe)
		if trans is not None:
			self.flow.append(trans)
		if debpe is not None:
			self.flow.append(debpe)
		if detruecaser is not None:
			self.flow.append(detruecaser)
		if detok is not None:
			self.flow.append(detok)

	def __call__(self, paragraph):

		_tmp = [tmpu.strip() for tmpu in paragraph.strip().split("\n")]
		_rs = []
		_tmpi = None
		if self.sent_split is not None:
			np = len(_tmp) - 1
			if np > 0:
				for _i, _tmpu in enumerate(_tmp):
					if _tmpu:
						_rs.extend(self.sent_split(_tmpu))
					if _i < np:
						_rs.append("")
				_tmpi = sorti(_rs)
				_tmp = _tmpi
			else:
				_tmp = [" ".join(clear_list(_tmp[0].split()))]
		else:
			_tmp = clean_list(_tmp)

		for pu in self.flow:
			_tmp = pu(_tmp)

		if len(_rs) > 1:
			_tmp = restore(_rs, _tmpi, _tmp)
			return "\n".join(_tmp)

		return " ".join(_tmp)
