#encoding: utf-8

import torch

from transformer.NMT import NMT
from transformer.EnsembleNMT import NMT as Ensemble
from parallel.parallelMT import DataParallelMT

from utils.base import *
from utils.fmt.base import ldvocab, clean_str
from utils.fmt.base4torch import parse_cuda_decode

from utils.fmt.single import batch_padder

def data_loader(sentences_iter, vcbi, minbsize=1, bsize=768, maxpad=16, maxpart=4, maxtoken=3920):
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
			data[_sl] = clean_str(_tl)

	rs = []
	_tl = []
	for line in src:
		tmp = line.strip()
		if tmp:
			tmp = clean_str(tmp)
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

		self.use_cuda, self.cuda_device, cuda_devices, self.multi_gpu = parse_cuda_decode(cnfg.use_cuda, cnfg.gpuid, cnfg.multi_gpu_decoding)

		if self.use_cuda:
			model.to(self.cuda_device)
			if self.multi_gpu:
				model = DataParallelMT(model, device_ids=cuda_devices, output_device=cuda_device.index, host_replicate=True, gather_output=False)

		self.beam_size = cnfg.beam_size

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
				_tmp = [clean_str(_tmp[0])]
		else:
			_tmp = [clean_str(tmpu) for tmpu in _tmp]

		for pu in self.flow:
			_tmp = pu(_tmp)

		if len(_rs) > 1:
			_tmp = restore(_rs, _tmpi, _tmp)
			return "\n".join(_tmp)

		return " ".join(_tmp)

#import cnfg
#from datautils.moses import SentenceSplitter, Tokenizer, Detokenizer, Normalizepunctuation, Truecaser, Detruecaser
#from datautils.bpe import BPEApplier, BPEApplier, BPERemover
#if __name__ == "__main__":
	#tl = ["28 @-@ jähriger Koch in San Francisco M@@ all tot a@@ u@@ f@@ gefunden", "ein 28 @-@ jähriger Koch , der vor kurzem nach San Francisco gezogen ist , wurde im T@@ r@@ e@@ p@@ p@@ e@@ n@@ haus eines örtlichen E@@ i@@ n@@ k@@ a@@ u@@ f@@ z@@ e@@ n@@ t@@ r@@ u@@ ms tot a@@ u@@ f@@ gefunden .", "der Bruder des O@@ p@@ f@@ e@@ r@@ s sagte aus , dass er sich niemanden vorstellen kann , der ihm schaden wollen würde , &quot; E@@ n@@ d@@ lich ging es bei ihm wieder b@@ e@@ r@@ g@@ auf &quot; .", "der am Mittwoch morgen in der W@@ e@@ s@@ t@@ field M@@ all g@@ e@@ f@@ u@@ n@@ d@@ e@@ n@@ e L@@ e@@ i@@ c@@ h@@ n@@ a@@ m wurde als der 28 Jahre alte Frank G@@ a@@ l@@ i@@ c@@ i@@ a aus San Francisco identifiziert , teilte die g@@ e@@ r@@ i@@ c@@ h@@ t@@ s@@ medizinische Abteilung in San Francisco mit .", "das San Francisco P@@ o@@ l@@ i@@ ce D@@ e@@ p@@ a@@ r@@ t@@ ment sagte , dass der Tod als Mord eingestuft wurde und die Ermittlungen am L@@ a@@ u@@ f@@ en sind .", "der Bruder des O@@ p@@ f@@ e@@ r@@ s , Louis G@@ a@@ l@@ i@@ c@@ i@@ a , teilte dem A@@ B@@ S Sender K@@ GO in San Francisco mit , dass Frank , der früher als Koch in B@@ o@@ s@@ t@@ on gearbeitet hat , vor sechs Monaten seinen T@@ r@@ a@@ u@@ m@@ j@@ ob als Koch im S@@ o@@ n@@ s &amp; D@@ a@@ u@@ g@@ h@@ t@@ e@@ r@@ s Restaurant in San Francisco e@@ r@@ g@@ a@@ t@@ t@@ e@@ r@@ t hatte .", "ein Sprecher des S@@ o@@ n@@ s &amp; D@@ a@@ u@@ g@@ h@@ t@@ e@@ r@@ s sagte , dass sie über seinen Tod &quot; s@@ c@@ h@@ o@@ c@@ k@@ i@@ e@@ r@@ t und am Boden zerstört seien &quot; .", "&quot; wir sind ein kleines Team , das wie eine enge Familie arbeitet und wir werden ihn s@@ c@@ h@@ m@@ e@@ r@@ z@@ lich vermissen &quot; , sagte der Sprecher weiter .", "unsere Gedanken und unser B@@ e@@ i@@ leid sind in dieser schweren Zeit bei F@@ r@@ a@@ n@@ k@@ s Familie und Freunden .", "Louis G@@ a@@ l@@ i@@ c@@ i@@ a gab an , dass Frank zunächst in Hostels lebte , aber dass , &quot; die Dinge für ihn endlich b@@ e@@ r@@ g@@ auf gingen &quot; ."]
	#spl = SentenceSplitter("de")
	#tok = Tokenizer("de")
	#detok = Detokenizer("en")
	#punc_norm = Normalizepunctuation("de")
	#truecaser = Truecaser("c1207\\truecase-model.de")
	#detruecaser = Detruecaser()
	#tran_core = TranslatorCore("c1207\\eva_20_1.384_1.088_26.74.t7", "c1207\\src.vcb", "c1207\\tgt.vcb", cnfg)
	#bpe = BPEApplier("c1207\\src.cds", "c1207\\src.vcb.bpe", 50)
	#debpe = BPERemover()
	#trans = Translator(tran_core, spl, tok, detok, bpe, debpe, punc_norm, truecaser, detruecaser)
	#rs = tran_core(tl)
	#for rsu in rs:
		#print(rsu.replace("@@ ", ""))
	#rs = trans(". ".join(detok(detruecaser(debpe(tl)))))
	#print(rs)
