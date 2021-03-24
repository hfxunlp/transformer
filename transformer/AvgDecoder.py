#encoding: utf-8

import torch
from torch import nn
from modules.base import *
from utils.sampler import SampleMax
from utils.base import all_done, index_tensors, expand_bsize_for_beam
from utils.aan import share_aan_cache
from math import sqrt

from utils.fmt.base import pad_id

from transformer.Decoder import DecoderLayer as DecoderLayerBase
from transformer.Decoder import Decoder as DecoderBase

# Average Decoder is proposed in Accelerating Neural Transformer via an Average Attention Network (https://www.aclweb.org/anthology/P18-1166/)

from cnfg.ihyp import *

class DecoderLayer(DecoderLayerBase):

	# isize: input size
	# fhsize: hidden size of PositionwiseFeedForward
	# attn_drop: dropout for MultiHeadAttention
	# num_head: number of heads in MultiHeadAttention
	# ahsize: hidden size of MultiHeadAttention

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, ahsize=None, **kwargs):

		_ahsize = isize if ahsize is None else ahsize
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(DecoderLayer, self).__init__(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, ahsize=_ahsize, **kwargs)

		self.self_attn = AverageAttn(isize, _fhsize, dropout)

	# inpute: encoded representation from encoder (bsize, seql, isize)
	# inputo: embedding of decoded translation (bsize, nquery, isize) during training, layer normed summed previous states for decoding
	# src_pad_mask: mask for given encoding source sentence (bsize, nquery, seql), see Encoder, expanded after generated with:
	#	src_pad_mask = input.eq(0).unsqueeze(1)
	# query_unit: single query to decode, used to support decoding for given step
	# step: current decoding step, used to average over the sum.

	def forward(self, inpute, inputo, src_pad_mask=None, query_unit=None, step=1):

		if query_unit is None:
			_inputo = self.layer_normer1(inputo)

			context = self.self_attn(_inputo, _inputo)

			if self.drop is not None:
				context = self.drop(context)

			context = context + (_inputo if self.norm_residual else inputo)
		else:
			_query_unit = self.layer_normer1(query_unit)

			if step == 1:
				states_return = _query_unit

				context = self.self_attn(_query_unit, _query_unit, True)
			else:
				states_return = inputo + _query_unit

				context = self.self_attn(_query_unit, states_return / float(step), True)

			if self.drop is not None:
				context = self.drop(context)

			context = context + (_query_unit if self.norm_residual else query_unit)

		_context = self.layer_normer2(context)
		_context_new = self.cross_attn(_context, inpute, mask=src_pad_mask)

		if self.drop is not None:
			_context_new = self.drop(_context_new)

		context = _context_new + (_context if self.norm_residual else context)

		context = self.ff(context)

		if query_unit is None:
			return context
		else:
			return context, states_return

class Decoder(DecoderBase):

	# construction function is needed, since DecoderLayer should be re-assigned.

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

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, emb_w=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindemb=True, forbidden_index=None, share_layer=False, **kwargs):

		_ahsize = isize if ahsize is None else ahsize
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Decoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindemb=bindemb, forbidden_index=forbidden_index, share_layer=share_layer, **kwargs)

		if share_layer:
			_shared_layer = DecoderLayer(isize, _fhsize, dropout, attn_drop, num_head, _ahsize)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([DecoderLayer(isize, _fhsize, dropout, attn_drop, num_head, _ahsize) for i in range(num_layer)])

		self.mask = None

		share_aan_cache(self)

	# inpute: encoded representation from encoder (bsize, seql, isize)
	# inputo: decoded translation (bsize, nquery)
	# src_pad_mask: mask for given encoding source sentence (bsize, 1, seql), see Encoder, generated with:
	#	src_pad_mask = input.eq(0).unsqueeze(1)

	def forward(self, inpute, inputo, src_pad_mask=None):

		bsize, nquery = inputo.size()

		out = self.wemb(inputo)

		out = out * sqrt(out.size(-1))
		if self.pemb is not None:
			out = out + self.pemb(inputo, expand=False)

		if self.drop is not None:
			out = self.drop(out)

		for net in self.nets:
			out = net(inpute, out, src_pad_mask)

		if self.out_normer is not None:
			out = self.out_normer(out)

		out = self.lsm(self.classifier(out))

		return out

	def load_base(self, base_decoder):

		self.drop = base_decoder.drop

		self.wemb = base_decoder.wemb

		self.pemb = base_decoder.pemb

		_nets = list(base_decoder.nets)

		self.nets = nn.ModuleList(_nets + list(self.nets[len(_nets):]))

		self.classifier = base_decoder.classifier

		self.lsm = base_decoder.lsm

		self.out_normer = None if self.out_normer is None else base_decoder.out_normer

	# inpute: encoded representation from encoder (bsize, seql, isize)
	# src_pad_mask: mask for given encoding source sentence (bsize, 1, seql), see Encoder, generated with:
	#	src_pad_mask = input.eq(0).unsqueeze(1)
	# max_len: maximum length to generate

	def greedy_decode(self, inpute, src_pad_mask=None, max_len=512, fill_pad=False, sample=False):

		bsize = inpute.size(0)

		sos_emb = self.get_sos_emb(inpute)

		sqrt_isize = sqrt(sos_emb.size(-1))

		# out: input to the decoder for the first step (bsize, 1, isize)

		out = sos_emb * sqrt_isize
		if self.pemb is not None:
			 out = out + self.pemb.get_pos(0)

		if self.drop is not None:
			out = self.drop(out)

		states = {}

		for _tmp, net in enumerate(self.nets):
			out, _state = net(inpute, None, src_pad_mask, out, 1)
			states[_tmp] = _state

		if self.out_normer is not None:
			out = self.out_normer(out)

		# out: (bsize, 1, nwd)
		out = self.classifier(out)
		# wds: (bsize, 1)
		wds = SampleMax(out.softmax(-1), dim=-1, keepdim=False) if sample else out.argmax(dim=-1)

		trans = [wds]

		# done_trans: (bsize, 1)

		done_trans = wds.eq(2)

		for i in range(2, max_len + 1):

			out = self.wemb(wds) * sqrt_isize
			if self.pemb is not None:
				out = out + self.pemb.get_pos(i - 1)

			if self.drop is not None:
				out = self.drop(out)

			for _tmp, net in enumerate(self.nets):
				out, _state = net(inpute, states[_tmp], src_pad_mask, out, i)
				states[_tmp] = _state

			if self.out_normer is not None:
				out = self.out_normer(out)

			# out: (bsize, 1, nwd)
			out = self.classifier(out)
			# wds: (bsize, 1)
			wds = SampleMax(out.softmax(-1), dim=-1, keepdim=False) if sample else out.argmax(dim=-1)

			trans.append(wds.masked_fill(done_trans, pad_id) if fill_pad else wds)

			done_trans = done_trans | wds.eq(2)
			if all_done(done_trans, bsize):
				break

		return torch.cat(trans, 1)

	# inpute: encoded representation from encoder (bsize, seql, isize)
	# src_pad_mask: mask for given encoding source sentence (bsize, 1, seql), see Encoder, generated with:
	#	src_pad_mask = input.eq(0).unsqueeze(1)
	# beam_size: beam size
	# max_len: maximum length to generate

	def beam_decode(self, inpute, src_pad_mask=None, beam_size=8, max_len=512, length_penalty=0.0, return_all=False, clip_beam=clip_beam_with_lp, fill_pad=False):

		bsize, seql = inpute.size()[:2]

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

		out = sos_emb * sqrt_isize
		if self.pemb is not None:
			 out = out + self.pemb.get_pos(0)

		if self.drop is not None:
			out = self.drop(out)

		states = {}

		for _tmp, net in enumerate(self.nets):
			out, _state = net(inpute, None, src_pad_mask, out, 1)
			states[_tmp] = _state

		if self.out_normer is not None:
			out = self.out_normer(out)

		# out: (bsize, 1, nwd)

		out = self.lsm(self.classifier(out))

		# scores: (bsize, 1, beam_size) => (bsize, beam_size)
		# wds: (bsize * beam_size, 1)
		# trans: (bsize * beam_size, 1)

		scores, wds = out.topk(beam_size, dim=-1)
		scores = scores.squeeze(1)
		sum_scores = scores
		wds = wds.view(real_bsize, 1)
		trans = wds

		# done_trans: (bsize, beam_size)

		done_trans = wds.view(bsize, beam_size).eq(2)

		# inpute: (bsize, seql, isize) => (bsize * beam_size, seql, isize)

		self.repeat_cross_attn_buffer(beam_size)

		# _src_pad_mask: (bsize, 1, seql) => (bsize * beam_size, 1, seql)

		_src_pad_mask = None if src_pad_mask is None else src_pad_mask.repeat(1, beam_size, 1).view(real_bsize, 1, seql)

		# states[i]: (bsize, 1, isize) => (bsize * beam_size, 1, isize)

		states = expand_bsize_for_beam(states, beam_size=beam_size)

		for step in range(2, max_len + 1):

			out = self.wemb(wds) * sqrt_isize
			if self.pemb is not None:
				 out = out + self.pemb.get_pos(step - 1)

			if self.drop is not None:
				out = self.drop(out)

			for _tmp, net in enumerate(self.nets):
				out, _state = net(inpute, states[_tmp], _src_pad_mask, out, step)
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

			_scores, _wds = out.topk(beam_size, dim=-1)
			_scores = (_scores.masked_fill(done_trans.unsqueeze(2).expand(bsize, beam_size, beam_size), 0.0) + sum_scores.unsqueeze(2).expand(bsize, beam_size, beam_size))

			if length_penalty > 0.0:
				lpv = lpv.masked_fill(1 - done_trans.view(real_bsize, 1), ((step + 5.0) ** length_penalty) / lpv_base)

			# clip from k ** 2 candidate and remain the top-k for each path
			# scores: (bsize, beam_size * beam_size) => (bsize, beam_size)
			# _inds: indexes for the top-k candidate (bsize, beam_size)

			if clip_beam and (length_penalty > 0.0):
				scores, _inds = (_scores.view(real_bsize, beam_size) / lpv.expand(real_bsize, beam_size)).view(bsize, beam_size2).topk(beam_size, dim=-1)
				_tinds = (_inds + torch.arange(0, bsizeb2, beam_size2, dtype=_inds.dtype, device=_inds.device).unsqueeze(1).expand_as(_inds)).view(real_bsize)
				sum_scores = _scores.view(bsizeb2).index_select(0, _tinds).view(bsize, beam_size)
			else:
				scores, _inds = _scores.view(bsize, beam_size2).topk(beam_size, dim=-1)
				_tinds = (_inds + torch.arange(0, bsizeb2, beam_size2, dtype=_inds.dtype, device=_inds.device).unsqueeze(1).expand_as(_inds)).view(real_bsize)
				sum_scores = scores

			# select the top-k candidate with higher route score and update translation record
			# wds: (bsize, beam_size, beam_size) => (bsize * beam_size, 1)

			wds = _wds.view(bsizeb2).index_select(0, _tinds).view(real_bsize, 1)

			# reduces indexes in _inds from (beam_size ** 2) to beam_size
			# thus the fore path of the top-k candidate is pointed out
			# _inds: indexes for the top-k candidate (bsize, beam_size)

			_inds = (_inds // beam_size + torch.arange(0, real_bsize, beam_size, dtype=_inds.dtype, device=_inds.device).unsqueeze(1).expand_as(_inds)).view(real_bsize)

			# select the corresponding translation history for the top-k candidate and update translation records
			# trans: (bsize * beam_size, nquery) => (bsize * beam_size, nquery + 1)

			trans = torch.cat((trans.index_select(0, _inds), wds.masked_fill(done_trans.view(real_bsize, 1), pad_id) if fill_pad else wds), 1)

			done_trans = (done_trans.view(real_bsize).index_select(0, _inds) | wds.eq(2).squeeze(1)).view(bsize, beam_size)

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

			# update the corresponding hidden states
			# states[i]: (bsize * beam_size, nquery, isize)
			# _inds: (bsize, beam_size) => (bsize * beam_size)

			states = index_tensors(states, indices=_inds, dim=0)

		# if length penalty is only applied in the last step, apply length penalty
		if (not clip_beam) and (length_penalty > 0.0):
			scores = scores / lpv.view(bsize, beam_size)
			scores, _inds = scores.topk(beam_size, dim=-1)
			_inds = (_inds + torch.arange(0, real_bsize, beam_size, dtype=_inds.dtype, device=_inds.device).unsqueeze(1).expand_as(_inds)).view(real_bsize)
			trans = trans.view(real_bsize, -1).index_select(0, _inds).view(bsize, beam_size, -1)

		if return_all:

			return trans, scores
		else:

			return trans.view(bsize, beam_size, -1).select(1, 0)
