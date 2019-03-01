#encoding: utf-8

import torch
from torch import nn

# import Encoder and Decoder from transformer.AGG.InceptEncoder and transformer.AGG.InceptDecoder/transformer.AGG.InceptAvgDecoder to learn complex representation with incepted transformer
from transformer.Encoder import Encoder

# switch the comment between the following two lines to choose standard decoder or average decoder
from transformer.Decoder import Decoder
#from transformer.AvgDecoder import Decoder

from math import sqrt

class NMT(nn.Module):

	# isize: size of word embedding
	# snwd: number of words for Encoder
	# tnwd: number of words for Decoder
	# num_layer: number of encoder layers
	# fhsize: number of hidden units for PositionwiseFeedForward
	# attn_drop: dropout for MultiHeadAttention
	# global_emb: Sharing the embedding between encoder and decoder, which means you should have a same vocabulary for source and target language
	# num_head: number of heads in MultiHeadAttention
	# xseql: maxmimum length of sequence
	# ahsize: number of hidden units for MultiHeadAttention

	def __init__(self, isize, snwd, tnwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, global_emb=False, num_head=8, xseql=512, ahsize=None, norm_output=True, bindDecoderEmb=False, forbidden_index=None):

		super(NMT, self).__init__()

		self.enc = Encoder(isize, snwd, num_layer, fhsize, dropout, attn_drop, num_head, xseql, ahsize, norm_output)

		emb_w = self.enc.wemb.weight if global_emb else None

		self.dec = Decoder(isize, tnwd, num_layer, fhsize, dropout, attn_drop, emb_w, num_head, xseql, ahsize, norm_output, bindDecoderEmb, forbidden_index)
		#self.dec = Decoder(isize, tnwd, num_layer + 2, dropout, attn_drop, emb_w, num_head, xseql, ahsize, norm_output, bindDecoderEmb, forbidden_index)

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

		bsize, _ = inpute.size()

		# out: input to the decoder for the first step (bsize, 1)

		out = inpute.new_ones(bsize, 1)

		done_trans = None

		for i in range(0, max_len):

			_out = self.dec(ence, out, mask)

			_out = torch.argmax(_out, dim=-1)

			wds = _out.narrow(1, _out.size(1) - 1, 1)

			out = torch.cat((out, wds), -1)

			# done_trans: (bsize)
			done_trans = wds.squeeze(1).eq(2) if done_trans is None else torch.gt(done_trans + wds.squeeze(1).eq(2), 0)

			if done_trans.sum().item() == bsize:
				break

		return out.narrow(1, 1, out.size(1) - 1)

	def train_beam_decode(self, inpute, mask=None, beam_size=8, max_len=512, length_penalty=0.0, return_all=False, clip_beam=False):

		bsize, seql = inpute.size()

		real_bsize = bsize * beam_size

		ence = self.enc(inpute, mask).repeat(1, beam_size, 1).view(real_bsize, seql, -1)

		mask = mask.repeat(1, beam_size, 1).view(real_bsize, 1, seql)

		# out: input to the decoder for the first step (bsize * beam_size, 1)

		out = inpute.new_ones(real_bsize, 1)

		if length_penalty > 0.0:
			# lpv: length penalty vector for each beam (bsize * beam_size, 1)
			lpv = ence.new_ones(real_bsize, 1)
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
			_scores, _wds = torch.topk(_out, beam_size, dim=-1)

			if done_trans is not None:
				_scores = _scores.masked_fill(done_trans.unsqueeze(2).expand(bsize, beam_size, beam_size), 0.0) + sum_scores.unsqueeze(2).expand(bsize, beam_size, beam_size)

				if length_penalty > 0.0:
					lpv = lpv.masked_fill(1 - done_trans.view(real_bsize, 1), ((step + 5.0) ** length_penalty) / lpv_base)

			# scores/_inds: (bsize, beam_size)
			if clip_beam and (length_penalty > 0.0):
				scores, _inds = torch.topk((_scores / lpv.expand(real_bsize, beam_size)).view(bsize, beam_size2), beam_size, dim=-1)
				_tinds = (_inds + torch.arange(0, bsizeb2, beam_size2, dtype=_inds.dtype, device=_inds.device).unsqueeze(1).expand_as(_inds)).view(real_bsize)

				# sum_scores: (bsize, beam_size)			
				sum_scores = _scores.view(bsizeb2).index_select(0, _tinds).view(bsize, beam_size)
				
			else:
				scores, _inds = torch.topk(_scores.view(bsize, beam_size2), beam_size, dim=-1)
				_tinds = (_inds + torch.arange(0, bsizeb2, beam_size2, dtype=_inds.dtype, device=_inds.device).unsqueeze(1).expand_as(_inds)).view(real_bsize)
				sum_scores = scores

			# wds: (bsize * beam_size, 1)
			wds = _wds.view(bsizeb2).index_select(0, _tinds).view(real_bsize, 1)

			_inds = (_inds / beam_size + torch.arange(0, real_bsize, beam_size, dtype=_inds.dtype, device=_inds.device).unsqueeze(1).expand_as(_inds)).view(real_bsize)
			out = torch.cat((out.index_select(0, _inds), wds), -1)

			# done_trans: (bsize, beam_size)
			done_trans = wds.view(bsize, beam_size).eq(2) if done_trans is None else torch.gt(done_trans.view(real_bsize).index_select(0, _inds) + wds.view(real_bsize).eq(2), 0).view(bsize, beam_size)

			# check early stop for beam search
			# done_trans: (bsize, beam_size)
			# scores: (bsize, beam_size)

			_done = False
			if length_penalty > 0.0:
				lpv = lpv.index_select(0, _inds)
			elif (not return_all) and done_trans.select(1, 0).sum().item() == bsize:
				_done = True

			# check beam states(done or not)

			if _done or (done_trans.sum().item() == real_bsize):
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
