#encoding: utf-8

import torch
from torch.cuda.amp import autocast

from transformer.NMT import NMT
from transformer.EnsembleNMT import NMT as Ensemble
from parallel.parallelMT import DataParallelMT

from utils.base import *
from utils.fmt.base import ldvocab, clean_str, reverse_dict, eos_id, clean_liststr_lentok, dict_insert_set, iter_dict_sort
from utils.fmt.base4torch import parse_cuda_decode

from utils.fmt.single import batch_padder

from cnfg.ihyp import *

def data_loader(sentences_iter, vcbi, minbsize=1, bsize=768, maxpad=16, maxpart=4, maxtoken=3920):
	for i_d in batch_padder(sentences_iter, vcbi, bsize, maxpad, maxpart, maxtoken, minbsize):
		yield torch.tensor(i_d, dtype=torch.long)

def load_fixing(module):
	if hasattr(module, "fix_load"):
		module.fix_load()

def sorti(lin):

	data = {}

	for ls in lin:
		ls = ls.strip()
		if ls:
			data = dict_insert_set(data, ls, len(ls.split()))

	return list(iter_dict_sort(data))

def restore(src, tsrc, trs):

	data = {}

	for sl, tl in zip(tsrc, trs):
		_sl, _tl = sl.strip(), tl.strip()
		if _sl and _tl:
			data[_sl] = clean_str(_tl)

	return [data.get(clean_str(line.strip()), line) for line in src]

class TranslatorCore:

	def __init__(self, modelfs, fvocab_i, fvocab_t, cnfg, minbsize=1, expand_for_mulgpu=True, bsize=64, maxpad=16, maxpart=4, maxtoken=1536, minfreq = False, vsize = False):

		vcbi, nwordi = ldvocab(fvocab_i, minf=minfreq, omit_vsize=vsize, vanilla=False)
		vcbt, nwordt = ldvocab(fvocab_t, minf=minfreq, omit_vsize=vsize, vanilla=False)
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
				tmp = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes)

				tmp = load_model_cpu(modelf, tmp)
				tmp.apply(load_fixing)

				models.append(tmp)
			model = Ensemble(models)

		else:
			model = NMT(cnfg.isize, nwordi, nwordt, cnfg.nlayer, cnfg.ff_hsize, cnfg.drop, cnfg.attn_drop, cnfg.share_emb, cnfg.nhead, cache_len_default, cnfg.attn_hsize, cnfg.norm_output, cnfg.bindDecoderEmb, cnfg.forbidden_indexes)

			model = load_model_cpu(modelfs, model)
			model.apply(load_fixing)

		model.eval()

		self.use_cuda, self.cuda_device, cuda_devices, self.multi_gpu = parse_cuda_decode(cnfg.use_cuda, cnfg.gpuid, cnfg.multi_gpu_decoding)

		if self.use_cuda:
			model.to(self.cuda_device)
			if self.multi_gpu:
				model = DataParallelMT(model, device_ids=cuda_devices, output_device=self.cuda_device.index, host_replicate=True, gather_output=False)
		self.use_amp = cnfg.use_amp and self.use_cuda

		self.beam_size = cnfg.beam_size

		self.length_penalty = cnfg.length_penalty
		self.net = model

	def __call__(self, sentences_iter):
		rs = []
		with torch.no_grad():
			for seq_batch in data_loader(sentences_iter, self.vcbi, self.minbsize, self.bsize, self.maxpad, self.maxpart, self.maxtoken):
				if self.use_cuda:
					seq_batch = seq_batch.to(self.cuda_device)
				with autocast(enabled=self.use_amp):
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
						if tmpu == eos_id:
							break
						else:
							tmp.append(self.vcbt[tmpu])
					rs.append(" ".join(tmp))
				seq_batch = None
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

	def __call__(self, paragraphs):

		_paras = [clean_str(tmpu.strip()) for tmpu in paragraphs.strip().split("\n") if tmpu]

		_tmp = []
		if self.sent_split is None:
			for _tmpu in paras:
				_tmp.append(_tmpu)
				_tmp.append("\n")
		else:
			for _tmpu in _paras:
				_tmp.extend(clean_list([clean_str(_tmps) for _tmps in self.sent_split(_tmpu)]))
				_tmp.append("\n")
		_tmp_o = _tmpi = sorti(_tmp)

		for pu in self.flow:
			_tmp_o = pu(_tmp_o)

		_tmp = restore(_tmp, _tmpi, _tmp_o)

		return " ".join(_tmp).replace(" \n", "\n").replace("\n ", "\n")

#import cnfg
#from datautils.moses import SentenceSplitter
#from datautils.pymoses import Tokenizer, Detokenizer, Normalizepunctuation, Truecaser, Detruecaser
#from datautils.bpe import BPEApplier, BPERemover
#if __name__ == "__main__":
	#tl = ["28 @-@ jähriger Koch in San Francisco M@@ all tot a@@ u@@ f@@ gefunden", "ein 28 @-@ jähriger Koch , der vor kurzem nach San Francisco gezogen ist , wurde im T@@ r@@ e@@ p@@ p@@ e@@ n@@ haus eines örtlichen E@@ i@@ n@@ k@@ a@@ u@@ f@@ z@@ e@@ n@@ t@@ r@@ u@@ ms tot a@@ u@@ f@@ gefunden .", "der Bruder des O@@ p@@ f@@ e@@ r@@ s sagte aus , dass er sich niemanden vorstellen kann , der ihm schaden wollen würde , &quot; E@@ n@@ d@@ lich ging es bei ihm wieder b@@ e@@ r@@ g@@ auf &quot; .", "der am Mittwoch morgen in der W@@ e@@ s@@ t@@ field M@@ all g@@ e@@ f@@ u@@ n@@ d@@ e@@ n@@ e L@@ e@@ i@@ c@@ h@@ n@@ a@@ m wurde als der 28 Jahre alte Frank G@@ a@@ l@@ i@@ c@@ i@@ a aus San Francisco identifiziert , teilte die g@@ e@@ r@@ i@@ c@@ h@@ t@@ s@@ medizinische Abteilung in San Francisco mit .", "das San Francisco P@@ o@@ l@@ i@@ ce D@@ e@@ p@@ a@@ r@@ t@@ ment sagte , dass der Tod als Mord eingestuft wurde und die Ermittlungen am L@@ a@@ u@@ f@@ en sind .", "der Bruder des O@@ p@@ f@@ e@@ r@@ s , Louis G@@ a@@ l@@ i@@ c@@ i@@ a , teilte dem A@@ B@@ S Sender K@@ GO in San Francisco mit , dass Frank , der früher als Koch in B@@ o@@ s@@ t@@ on gearbeitet hat , vor sechs Monaten seinen T@@ r@@ a@@ u@@ m@@ j@@ ob als Koch im S@@ o@@ n@@ s &amp; D@@ a@@ u@@ g@@ h@@ t@@ e@@ r@@ s Restaurant in San Francisco e@@ r@@ g@@ a@@ t@@ t@@ e@@ r@@ t hatte .", "ein Sprecher des S@@ o@@ n@@ s &amp; D@@ a@@ u@@ g@@ h@@ t@@ e@@ r@@ s sagte , dass sie über seinen Tod &quot; s@@ c@@ h@@ o@@ c@@ k@@ i@@ e@@ r@@ t und am Boden zerstört seien &quot; .", "&quot; wir sind ein kleines Team , das wie eine enge Familie arbeitet und wir werden ihn s@@ c@@ h@@ m@@ e@@ r@@ z@@ lich vermissen &quot; , sagte der Sprecher weiter .", "unsere Gedanken und unser B@@ e@@ i@@ leid sind in dieser schweren Zeit bei F@@ r@@ a@@ n@@ k@@ s Familie und Freunden .", "Louis G@@ a@@ l@@ i@@ c@@ i@@ a gab an , dass Frank zunächst in Hostels lebte , aber dass , &quot; die Dinge für ihn endlich b@@ e@@ r@@ g@@ auf gingen &quot; ."]
	#spl = SentenceSplitter("de")
	#tok = Tokenizer("de")
	#detok = Detokenizer("en")
	#punc_norm = Normalizepunctuation("de")
	#truecaser = Truecaser("c1207\\truecase-model.de")
	#detruecaser = Detruecaser()
	#tran_core = TranslatorCore("c1207\\eva_20_1.384_1.088_26.74.h5", "c1207\\src.vcb", "c1207\\tgt.vcb", cnfg)
	#bpe = BPEApplier("c1207\\src.cds", "c1207\\src.vcb.bpe", 50)
	#debpe = BPERemover()
	#trans = Translator(tran_core, spl, tok, detok, bpe, debpe, punc_norm, truecaser, detruecaser)
	#rs = tran_core(tl)
	#for rsu in rs:
		#print(rsu.replace("@@ ", ""))
	#rs = trans(". ".join(detok(detruecaser(debpe(tl)))))
	#print(rs)
