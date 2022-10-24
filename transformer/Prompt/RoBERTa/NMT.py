#encoding: utf-8

import torch
from modules.base import Linear
from cnfg.vocab.plm.roberta import pad_id, mask_id

from transformer.PLM.RoBERTa.NMT import NMT as NMTBase

from cnfg.plm.roberta.ihyp import *

class NMT(NMTBase):

	def forward(self, inpute, token_types=None, mask=None, word_prediction=True):

		_mask = inpute.eq(pad_id).unsqueeze(1) if mask is None else mask
		out = self.enc(inpute, token_types=token_types, mask=_mask)
		_bsize, _, _hsize = out.size()
		_pm = inpute.eq(mask_id).unsqueeze(-1).expand(-1, -1, _hsize)

		return self.dec(out[_pm].view(_bsize, _hsize), word_prediction=word_prediction)
