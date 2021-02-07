#encoding: utf-8

import torch
from torch import nn

from utils.base import all_done

from transformer.EnsembleEncoder import Encoder

# switch the comment between the following two lines to choose standard decoder or average decoder
from transformer.EnsembleDecoder import Decoder
#from transformer.EnsembleAvgDecoder import Decoder

from cnfg.ihyp import *

class NMT(nn.Module):

	def __init__(self, models):

		super(NMT, self).__init__()

		self.enc = Encoder([model.enc for model in models])
		self.dec = Decoder([model.dec for model in models])

	# inpute: source sentences from encoder (bsize, seql)
	# inputo: decoded translation (bsize, nquery)
	# mask: user specified mask, otherwise it will be:
	#	inpute.eq(0).unsqueeze(1)

	def forward(self, inpute, inputo, mask=None):

		_mask = inpute.eq(0).unsqueeze(1) if mask is None else mask

		return self.dec(self.enc(inpute, _mask), inputo, _mask)

	# inpute: source sentences from encoder (bsize, seql)
	# beam_size: the beam size for beam search
	# max_len: maximum length to generate

	def decode(self, inpute, beam_size=1, max_len=None, length_penalty=0.0):

		mask = inpute.eq(0).unsqueeze(1)

		_max_len = inpute.size(1) + max(64, inpute.size(1) // 4) if max_len is None else max_len

		return self.dec.decode(self.enc(inpute, mask), mask, beam_size, _max_len, length_penalty)

	def train_decode(self, inpute, beam_size=1, max_len=None, length_penalty=0.0, mask=None):

		_mask = inpute.eq(0).unsqueeze(1) if mask is None else mask

		_max_len = inpute.size(1) + max(64, inpute.size(1) // 4) if max_len is None else max_len

		return self.train_beam_decode(inpute, _mask, beam_size, _max_len, length_penalty) if beam_size > 1 else self.train_greedy_decode(inpute, _mask, _max_len)

	def train_greedy_decode(self, inpute, mask=None, max_len=512):

		ence = self.enc(inpute, mask)

		bsize = inpute.size(0)

		# out: input to the decoder for the first step (bsize, 1)

		out = inpute.new_full((bsize, 1), 3)

		done_trans = None

		for i in range(0, max_len):

			_out = self.dec(ence, out, mask)

			_out = _out.argmax(dim=-1)

			wds = _out.narrow(1, _out.size(1) - 1, 1)

			out = torch.cat((out, wds), -1)

			# done_trans: (bsize)
			done_trans = wds.squeeze(1).eq(2) if done_trans is None else (done_trans | wds.squeeze(1).eq(2))

			if all_done(done_trans, bsize):
				break

		return out.narrow(1, 1, out.size(1) - 1)

	def train_beam_decode(self, inpute, mask=None, beam_size=8, max_len=512, length_penalty=0.0, return_all=False, clip_beam=clip_beam_with_lp):

		bsize, seql = inpute.size()

		real_bsize = bsize * beam_size

		ence = [encu.repeat(1, beam_size, 1).view(real_bsize, seql, -1) for encu in self.enc(inpute, mask)]

		mask = mask.repeat(1, beam_size, 1).view(real_bsize, 1, seql)

		# out: input to the decoder for the first step (bsize * beam_size, 1)

		out = inpute.new_full((real_bsize, 1), 3)

		if length_penalty > 0.0:
			# lpv: length penalty vector for each beam (bsize * beam_size, 1)
			lpv = ence[0].new_ones(real_bsize, 1)
			lpv_base = 6.0 ** length_penalty

		done_trans = None
		scores = None
		sum_scores = None

		beam_size2 = beam_size * beam_size
		bsizeb2 = bsize * beam_size2

		for step in range(1, max_len + 1):

			_out = self.dec(ence, out, mask)

			# _out: (bsize * beam_size, nquery, vocab_size) => (bsize, beam_size, vocab_size)
			_out = _out.narrow(1, _out.size(1) - 1, 1).view(bsize, beam_size, -1)

			# _scores/_wds: (bsize, beam_size, beam_size)
			_scores, _wds = _out.topk(beam_size, dim=-1)

			if done_trans is not None:
				_scores = _scores.masked_fill(done_trans.unsqueeze(2).expand(bsize, beam_size, beam_size), 0.0) + sum_scores.unsqueeze(2).expand(bsize, beam_size, beam_size)

				if length_penalty > 0.0:
					lpv = lpv.masked_fill(1 - done_trans.view(real_bsize, 1), ((step + 5.0) ** length_penalty) / lpv_base)

			# scores/_inds: (bsize, beam_size)
			if clip_beam and (length_penalty > 0.0):
				scores, _inds = (_scores / lpv.expand(real_bsize, beam_size)).view(bsize, beam_size2).topk(beam_size, dim=-1)
				_tinds = (_inds + torch.arange(0, bsizeb2, beam_size2, dtype=_inds.dtype, device=_inds.device).unsqueeze(1).expand_as(_inds)).view(real_bsize)

				# sum_scores: (bsize, beam_size)
				sum_scores = _scores.view(bsizeb2).index_select(0, _tinds).view(bsize, beam_size)

			else:
				scores, _inds = _scores.view(bsize, beam_size2).topk(beam_size, dim=-1)
				_tinds = (_inds + torch.arange(0, bsizeb2, beam_size2, dtype=_inds.dtype, device=_inds.device).unsqueeze(1).expand_as(_inds)).view(real_bsize)
				sum_scores = scores

			# wds: (bsize * beam_size, 1)
			wds = _wds.view(bsizeb2).index_select(0, _tinds).view(real_bsize, 1)

			_inds = (_inds // beam_size + torch.arange(0, real_bsize, beam_size, dtype=_inds.dtype, device=_inds.device).unsqueeze(1).expand_as(_inds)).view(real_bsize)
			out = torch.cat((out.index_select(0, _inds), wds), -1)

			# done_trans: (bsize, beam_size)
			done_trans = wds.view(bsize, beam_size).eq(2) if done_trans is None else (done_trans.view(real_bsize).index_select(0, _inds) | wds.view(real_bsize).eq(2)).view(bsize, beam_size)

			# check early stop for beam search
			# done_trans: (bsize, beam_size)
			# scores: (bsize, beam_size)

			_done = False
			if length_penalty > 0.0:
				lpv = lpv.index_select(0, _inds)
			elif (not return_all) and all_done(done_trans.select(1, 0), bsize):
				_done = True

			# check beam states(done or not)

			if _done or all_done(done_trans, real_bsize):
				break

		out = out.narrow(1, 1, out.size(1) - 1)

		# if length penalty is only applied in the last step, apply length penalty
		if (not clip_beam) and (length_penalty > 0.0):
			scores = scores / lpv.view(bsize, beam_size)

		if return_all:

			return out.view(bsize, beam_size, -1), scores
		else:

			# out: (bsize * beam_size, nquery) => (bsize, nquery)
			return out.view(bsize, beam_size, -1).select(1, 0)
