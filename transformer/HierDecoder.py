#encoding: utf-8

import torch
from torch import nn
from modules import *
from math import sqrt

from transformer.Decoder import DecoderLayer as DecoderLayerBase
from transformer.Decoder import Decoder as DecoderBase

class DecoderLayer(nn.Module):

	# isize: input size
	# fhsize: hidden size of PositionwiseFeedForward
	# attn_drop: dropout for MultiHeadAttention
	# num_head: number of heads in MultiHeadAttention
	# ahsize: hidden size of MultiHeadAttention

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, ahsize=None, num_sub=1, comb_input=True):

		_ahsize = isize if ahsize is None else ahsize

		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(DecoderLayer, self).__init__()

		self.nets = nn.ModuleList([DecoderLayerBase(isize, _fhsize, dropout, attn_drop, num_head, _ahsize) for i in range(num_sub)])

		self.combiner = ResidueCombiner(isize, num_sub + 1 if comb_input else num_sub, _fhsize)

		self.comb_input = comb_input

	# inpute: encoded representation from encoder (bsize, seql, isize)
	# inputo: embedding of decoded translation (bsize, nquery, isize)
	# src_pad_mask: mask for given encoding source sentence (bsize, nquery, seql), see Encoder, expanded after generated with:
	#	src_pad_mask = input.eq(0).unsqueeze(1)
	# tgt_pad_mask: mask to hide the future input
	# query_unit: single query to decode, used to support decoding for given step

	def forward(self, inpute, inputo, src_pad_mask=None, tgt_pad_mask=None, query_unit=None, concat_query=False):

		outs = []
		if query_unit is None:
			out = inputo
			if self.comb_input:
				outs.append(out)
			states_return = None
			for net in self.nets:
				out = net(inpute, out, src_pad_mask, tgt_pad_mask)
				outs.append(out)
		else:
			out = query_unit
			if self.comb_input:
				outs.append(out)
			states_return = []
			for _tmp, net in enumerate(self.nets):
				out, _state = net(inpute, None if inputo is None else inputo.select(-2, _tmp), src_pad_mask, tgt_pad_mask, out, concat_query)
				outs.append(out)
				states_return.append(_state)

			states_return = torch.stack(states_return, -2)

		out = self.combiner(*outs)

		if states_return is None:
			return out
		else:
			return out, states_return

class Decoder(DecoderBase):

	# isize: size of word embedding
	# nwd: number of words
	# num_layer: number of encoder layers
	# fhsize: number of hidden units for PositionwiseFeedForward
	# attn_drop: dropout for MultiHeadAttention
	# emb_w: weight for embedding. Use only when the encoder and decoder share a same dictionary
	# num_head: number of heads in MultiHeadAttention
	# xseql: maxmimum length of sequence
	# ahsize: number of hidden units for MultiHeadAttention
	# bindemb: bind embedding and classifier weight

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, emb_w=None, num_head=8, xseql=512, ahsize=None, norm_output=False, bindemb=False, forbidden_index=None, num_sub=1):

		_ahsize = isize if ahsize is None else ahsize

		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Decoder, self).__init__(isize, nwd, num_layer, _fhsize, dropout, attn_drop, emb_w, num_head, xseql, _ahsize, norm_output, bindemb, forbidden_index)

		self.nets = nn.ModuleList([DecoderLayer(isize, _fhsize, dropout, attn_drop, num_head, _ahsize, num_sub, i != 0) for i in range(num_layer)])

	# inpute: encoded representation from encoder (bsize, seql, isize)
	# src_pad_mask: mask for given encoding source sentence (bsize, 1, seql), see Encoder, generated with:
	#	src_pad_mask = input.eq(0).unsqueeze(1)
	# beam_size: beam size
	# max_len: maximum length to generate

	def beam_decode(self, inpute, src_pad_mask=None, beam_size=8, max_len=512, length_penalty=0.0, return_all=False, clip_beam=False):

		bsize, seql, _ = inpute.size()

		beam_size2 = beam_size * beam_size
		bsizeb2 = bsize * beam_size2
		real_bsize = bsize * beam_size

		sos_emb = self.get_sos_emb(inpute)
		isize = sos_emb.size(-1)
		sqrt_isize = sqrt(isize)

		if length_penalty > 0.0:
			# lpv: length penalty vector for each beam (bsize * beam_size, 1)
			lpv = sos_emb.new_ones(real_bsize, 1)
			lpv_base = 6.0 ** length_penalty

		out = sos_emb * sqrt_isize + self.pemb.get_pos(0).view(1, 1, isize).expand(bsize, 1, isize)

		if self.drop is not None:
			out = self.drop(out)

		states = {}

		for _tmp, net in enumerate(self.nets):
			out, _state = net(inpute, None, src_pad_mask, None, out, True)
			states[_tmp] = _state

		if self.out_normer is not None:
			out = self.out_normer(out)

		# out: (bsize, 1, nwd)

		out = self.lsm(self.classifier(out))

		# scores: (bsize, 1, beam_size) => (bsize, beam_size)
		# wds: (bsize * beam_size, 1)
		# trans: (bsize * beam_size, 1)

		scores, wds = torch.topk(out, beam_size, dim=-1)
		scores = scores.squeeze(1)
		sum_scores = scores
		wds = wds.view(real_bsize, 1)
		trans = wds

		# done_trans: (bsize, beam_size)

		done_trans = wds.view(bsize, beam_size).eq(2)

		# inpute: (bsize, seql, isize) => (bsize * beam_size, seql, isize)

		inpute = inpute.repeat(1, beam_size, 1).view(real_bsize, seql, isize)

		# _src_pad_mask: (bsize, 1, seql) => (bsize * beam_size, 1, seql)

		_src_pad_mask = None if src_pad_mask is None else src_pad_mask.repeat(1, beam_size, 1).view(real_bsize, 1, seql)

		# states[i]: (bsize, 1, num_sub, isize) => (bsize * beam_size, 1, num_sub, isize)

		for key, value in states.items():
			states[key] = value.repeat(1, beam_size, 1, 1).view(real_bsize, 1, -1, isize)

		for step in range(1, max_len):

			out = self.wemb(wds) * sqrt_isize + self.pemb.get_pos(step).view(1, 1, isize).expand(real_bsize, 1, isize)

			if self.drop is not None:
				out = self.drop(out)

			for _tmp, net in enumerate(self.nets):
				out, _state = net(inpute, states[_tmp], _src_pad_mask, None, out, True)
				states[_tmp] = _state

			if self.out_normer is not None:
				out = self.out_normer(out)

			# out: (bsize, beam_size, nwd)

			out = self.lsm(self.classifier(out)).view(bsize, beam_size, -1)

			# find the top k ** 2 candidates and calculate route scores for them
			# _scores: (bsize, beam_size, beam_size)
			# done_trans: (bsize, beam_size)
			# scores: (bsize, beam_size)
			# _wds: (bsize, beam_size, beam_size)
			# mask_from_done_trans: (bsize, beam_size) => (bsize, beam_size * beam_size)
			# added_scores: (bsize, 1, beam_size) => (bsize, beam_size, beam_size)

			_scores, _wds = torch.topk(out, beam_size, dim=-1)
			_scores = (_scores.masked_fill(done_trans.unsqueeze(2).expand(bsize, beam_size, beam_size), 0.0) + sum_scores.unsqueeze(2).expand(bsize, beam_size, beam_size))

			if length_penalty > 0.0:
				lpv = lpv.masked_fill(1 - done_trans.view(real_bsize, 1), ((step + 6.0) ** length_penalty) / lpv_base)

			# clip from k ** 2 candidate and remain the top-k for each path
			# scores: (bsize, beam_size * beam_size) => (bsize, beam_size)
			# _inds: indexes for the top-k candidate (bsize, beam_size)

			if clip_beam and (length_penalty > 0.0):
				scores, _inds = torch.topk((_scores.view(real_bsize, beam_size) / lpv.expand(real_bsize, beam_size)).view(bsize, beam_size2), beam_size, dim=-1)
				_tinds = (_inds + torch.arange(0, bsizeb2, beam_size2, dtype=_inds.dtype, device=_inds.device).unsqueeze(1).expand_as(_inds)).view(real_bsize)
				sum_scores = _scores.view(bsizeb2).index_select(0, _tinds).view(bsize, beam_size)
			else:
				scores, _inds = torch.topk(_scores.view(bsize, beam_size2), beam_size, dim=-1)
				_tinds = (_inds + torch.arange(0, bsizeb2, beam_size2, dtype=_inds.dtype, device=_inds.device).unsqueeze(1).expand_as(_inds)).view(real_bsize)
				sum_scores = scores

			# select the top-k candidate with higher route score and update translation record
			# wds: (bsize, beam_size, beam_size) => (bsize * beam_size, 1)

			wds = _wds.view(bsizeb2).index_select(0, _tinds).view(real_bsize, 1)

			# reduces indexes in _inds from (beam_size ** 2) to beam_size
			# thus the fore path of the top-k candidate is pointed out
			# _inds: indexes for the top-k candidate (bsize, beam_size)

			_inds = (_inds / beam_size + torch.arange(0, real_bsize, beam_size, dtype=_inds.dtype, device=_inds.device).unsqueeze(1).expand_as(_inds)).view(real_bsize)

			# select the corresponding translation history for the top-k candidate and update translation records
			# trans: (bsize * beam_size, nquery) => (bsize * beam_size, nquery + 1)

			trans = torch.cat((trans.index_select(0, _inds), wds), 1)

			done_trans = torch.gt(done_trans.view(real_bsize).index_select(0, _inds) + wds.eq(2).squeeze(1), 0).view(bsize, beam_size)

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

			# update the corresponding hidden states
			# states[i]: (bsize * beam_size, nquery, num_sub, isize)
			# _inds: (bsize, beam_size) => (bsize * beam_size)

			for key, value in states.items():
				states[key] = value.index_select(0, _inds)

		# if length penalty is only applied in the last step, apply length penalty
		if (not clip_beam) and (length_penalty > 0.0):
			scores = scores / lpv.view(bsize, beam_size)

		if return_all:

			return trans, scores
		else:

			return trans.view(bsize, beam_size, -1).select(1, 0)

	def load_base(self, base_decoder):

		self.drop = base_decoder.drop

		self.wemb = base_decoder.wemb

		self.pemb = base_decoder.pemb

		_nets = base_decoder.nets

		_lind = 0
		for net in self.nets:
			_rind = _lind + len(net.nets)
			net.nets = nn.ModuleList(_nets[_lind:_rind])
			_lind = _rind

		self.classifier = base_decoder.classifier

		self.lsm = base_decoder.lsm

		self.out_normer = None if self.out_normer is None else base_decoder.out_normer
		self.nets[-1].combiner.out_normer = base_decoder.out_normer
