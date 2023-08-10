#encoding: utf-8

from transformer.PLM.RoBERTa.NMT import NMT as NMTBase

from cnfg.plm.roberta.ihyp import *
from cnfg.vocab.plm.roberta import mask_id, pad_id

class NMT(NMTBase):

	def forward(self, inpute, token_types=None, mask=None, word_prediction=True, **kwargs):

		_mask = inpute.eq(pad_id).unsqueeze(1) if mask is None else mask
		out = self.enc(inpute, token_types=token_types, mask=_mask)
		_bsize, _, _hsize = out.size()

		return self.dec(out[inpute.eq(mask_id)].view(_bsize, _hsize), word_prediction=word_prediction)
