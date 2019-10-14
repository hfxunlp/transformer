#encoding: utf-8

import torch
from torch import nn

from modules.base import SelfAttn, PositionwiseFF, Linear, Dropout
from modules.__no_significance__.ua_cattn import CrossAttn

from utils.base import repeat_bsize_for_beam_tensor

from math import sqrt

from transformer.Decoder import Decoder as DecoderBase

class DecoderLayer(nn.Module):

	# isize: input size
	# fhsize: hidden size of PositionwiseFeedForward
	# attn_drop: dropout for MultiHeadAttention
	# num_head: number of heads in MultiHeadAttention
	# ahsize: hidden size of MultiHeadAttention
	# norm_residue: residue with layer normalized representation

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, ahsize=None, norm_residue=False):

		super(DecoderLayer, self).__init__()

		_ahsize = isize if ahsize is None else ahsize

		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		self.self_attn = SelfAttn(isize, _ahsize, isize, num_head, dropout=attn_drop)
		self.cross_attn = CrossAttn(isize, _ahsize, isize, num_head, dropout=attn_drop)

		self.ff = PositionwiseFF(isize, _fhsize, dropout, norm_residue)

		self.layer_normer1 = nn.LayerNorm(isize, eps=1e-06)
		self.layer_normer2 = nn.LayerNorm(isize, eps=1e-06)

		if dropout > 0:
			self.d1 = Dropout(dropout, inplace=True)
			self.d2 = Dropout(dropout, inplace=True)
		else:
			self.d1 = None
			self.d2 = None

		self.norm_residue = norm_residue

	# inpute: encoded representation from encoder (bsize, seql, isize)
	# inputo: embedding of decoded translation (bsize, nquery, isize)
	# src_pad_mask: mask for given encoding source sentence (bsize, nquery, seql), see Encoder, expanded after generated with:
	#	src_pad_mask = input.eq(0).unsqueeze(1)
	# tgt_pad_mask: mask to hide the future input
	# query_unit: single query to decode, used to support decoding for given step

	def forward(self, inpute, inputo, src_pad_mask=None, tgt_pad_mask=None, query_unit=None, concat_query=False):

		if query_unit is None:
			_inputo = self.layer_normer1(inputo)

			states_return = None

			context = self.self_attn(_inputo, mask=tgt_pad_mask)

			if self.d1 is not None:
				context = self.d1(context)

			context = context + (_inputo if self.norm_residue else inputo)

		else:
			_query_unit = self.layer_normer1(query_unit)

			if concat_query:

				_inputo = _query_unit if inputo is None else torch.cat((inputo, _query_unit), 1)

			else:
				_inputo = self.layer_normer1(inputo)

			states_return = _inputo

			context = self.self_attn(_query_unit, iK=_inputo)

			if self.d1 is not None:
				context = self.d1(context)

			context = context + (_query_unit if self.norm_residue else query_unit)

		_context = self.layer_normer2(context)
		_context_new, _attn = self.cross_attn(_context, inpute, mask=src_pad_mask)

		if self.d2 is not None:
			_context_new = self.d2(_context_new)

		context = _context_new + (_context if self.norm_residue else context)

		context = self.ff(context)

		if states_return is None:
			return context, _attn
		else:
			return context, _attn, states_return

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

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, emb_w=None, num_head=8, xseql=512, ahsize=None, norm_output=True, bindemb=False, forbidden_index=None):


		_ahsize = isize if ahsize is None else ahsize

		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Decoder, self).__init__(isize, nwd, num_layer, _fhsize, dropout, attn_drop, emb_w, num_head, xseql, _ahsize, norm_output, bindemb, forbidden_index)

		self.nets = nn.ModuleList([DecoderLayer(isize, _fhsize, dropout, attn_drop, num_head, _ahsize) for i in range(num_layer)])

		self.tattn_w = nn.Parameter(torch.Tensor(num_layer * num_head).uniform_(- sqrt(1.0 / (num_layer * num_head)), sqrt(1.0 / (num_layer * num_head))))
		self.tattn_drop = Dropout(dropout) if dropout > 0.0 else None

		self.classifier = nn.Sequential(Linear(isize * 2, isize, bias=False), Linear(isize, nwd))
		# be careful since this line of code is trying to share the weight of the wemb and the classifier, which may cause problems if torch.nn updates
		if bindemb:
			list(self.classifier.modules())[-1].weight = self.wemb.weight

	# inpute: encoded representation from encoder (bsize, seql, isize)
	# inputo: decoded translation (bsize, nquery)
	# src_pad_mask: mask for given encoding source sentence (bsize, 1, seql), see Encoder, generated with:
	#	src_pad_mask = input.eq(0).unsqueeze(1)

	def forward(self, inpute, inputo, src_pad_mask=None):

		bsize, nquery = inputo.size()

		out = self.wemb(inputo)

		out = out * sqrt(out.size(-1)) + self.pemb(inputo, expand=False)

		if self.drop is not None:
			out = self.drop(out)

		_mask = self._get_subsequent_mask(nquery)

		# the following line of code is to mask <pad> for the decoder,
		# which I think is useless, since only <pad> may pay attention to previous <pad> tokens, whos loss will be omitted by the loss function.
		#_mask = torch.gt(_mask + inputo.eq(0).unsqueeze(1), 0)

		attns = []
		for net in self.nets:
			out, _attn = net(inpute, out, src_pad_mask, _mask)
			attns.append(_attn)

		if self.out_normer is not None:
			out = self.out_normer(out)

		# attns: (bsize, num_layer * nheads, nquery, seql) => (bsize, nquery, seql, num_layer * nheads)
		attns = torch.cat(attns, dim=1).permute(0, 2, 3, 1)
		_asize = attns.size()

		# inpute: (bsize, seql, isize)
		# attns: (bsize, nquery, seql, num_layer * nheads) => (bsize, nquery, seql)
		# out: (bsize, nquery, isize * 2)
		out = torch.cat([out, attns.contiguous().view(-1, _asize[-1]).mv(self.tattn_w.softmax(dim=0) if self.tattn_drop is None else self.tattn_drop(self.tattn_w).softmax(dim=0)).view(_asize[:-1]).bmm(inpute)], dim=-1)

		out = self.lsm(self.classifier(out))

		return out

	# inpute: encoded representation from encoder (bsize, seql, isize)
	# src_pad_mask: mask for given encoding source sentence (bsize, 1, seql), see Encoder, generated with:
	#	src_pad_mask = input.eq(0).unsqueeze(1)
	# max_len: maximum length to generate

	def greedy_decode(self, inpute, src_pad_mask=None, max_len=512):

		bsize, seql = inpute.size()[:2]

		sos_emb = self.get_sos_emb(inpute)

		sqrt_isize = sqrt(sos_emb.size(-1))

		# out: input to the decoder for the first step (bsize, 1, isize)

		out = sos_emb * sqrt_isize + self.pemb.get_pos(0)

		if self.drop is not None:
			out = self.drop(out)

		states = {}

		attns = []
		for _tmp, net in enumerate(self.nets):
			out, _attn, _state = net(inpute, None, src_pad_mask, None, out, True)
			states[_tmp] = _state
			attns.append(_attn)

		if self.out_normer is not None:
			out = self.out_normer(out)

		attns = torch.cat(attns, dim=1).permute(0, 2, 3, 1)
		_asize = attns.size()
		out = torch.cat([out, attns.contiguous().view(-1, _asize[-1]).mv(self.tattn_w.softmax(dim=0) if self.tattn_drop is None else self.tattn_drop(self.tattn_w).softmax(dim=0)).view(_asize[:-1]).bmm(inpute)], dim=-1)

		# out: (bsize, 1, nwd)

		out = self.lsm(self.classifier(out))

		# wds: (bsize, 1)

		wds = out.argmax(dim=-1)

		trans = [wds]

		# done_trans: (bsize)

		done_trans = wds.squeeze(1).eq(2)

		for i in range(1, max_len):

			out = self.wemb(wds) * sqrt_isize + self.pemb.get_pos(i)

			if self.drop is not None:
				out = self.drop(out)

			attns = []
			for _tmp, net in enumerate(self.nets):
				out, _attn, _state = net(inpute, states[_tmp], src_pad_mask, None, out, True)
				states[_tmp] = _state
				attns.append(_attn)

			if self.out_normer is not None:
				out = self.out_normer(out)

			out = torch.cat([out, torch.cat(attns, dim=1).permute(0, 2, 3, 1).contiguous().view(-1, _asize[-1]).mv(self.tattn_w.softmax(dim=0) if self.tattn_drop is None else self.tattn_drop(self.tattn_w).softmax(dim=0)).view(_asize[:-1]).bmm(inpute)], dim=-1)

			# out: (bsize, 1, nwd)
			out = self.lsm(self.classifier(out))
			wds = out.argmax(dim=-1)

			trans.append(wds)

			done_trans = (done_trans + wds.squeeze(1).eq(2)).gt(0)
			if done_trans.sum().item() == bsize:
				break

		return torch.cat(trans, 1)

	# inpute: encoded representation from encoder (bsize, seql, isize)
	# src_pad_mask: mask for given encoding source sentence (bsize, 1, seql), see Encoder, generated with:
	#	src_pad_mask = input.eq(0).unsqueeze(1)
	# beam_size: beam size
	# max_len: maximum length to generate

	def beam_decode(self, inpute, src_pad_mask=None, beam_size=8, max_len=512, length_penalty=0.0, return_all=False, clip_beam=False):

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

		out = sos_emb * sqrt_isize + self.pemb.get_pos(0)

		if self.drop is not None:
			out = self.drop(out)

		states = {}

		attns = []
		for _tmp, net in enumerate(self.nets):
			out, _attn, _state = net(inpute, None, src_pad_mask, None, out, True)
			states[_tmp] = _state
			attns.append(_attn)

		if self.out_normer is not None:
			out = self.out_normer(out)

		attns = torch.cat(attns, dim=1).permute(0, 2, 3, 1)
		_asize = attns.size()
		out = torch.cat([out, attns.contiguous().view(-1, _asize[-1]).mv(self.tattn_w.softmax(dim=0) if self.tattn_drop is None else self.tattn_drop(self.tattn_w).softmax(dim=0)).view(_asize[:-1]).bmm(inpute)], dim=-1)

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

		inpute = inpute.repeat(1, beam_size, 1).view(real_bsize, seql, isize)

		# _src_pad_mask: (bsize, 1, seql) => (bsize * beam_size, 1, seql)

		_src_pad_mask = None if src_pad_mask is None else src_pad_mask.repeat(1, beam_size, 1).view(real_bsize, 1, seql)

		# states[i]: (bsize, 1, isize) => (bsize * beam_size, 1, isize)

		for key, value in states.items():
			states[key] = repeat_bsize_for_beam_tensor(value, beam_size)

		for step in range(1, max_len):

			out = self.wemb(wds) * sqrt_isize + self.pemb.get_pos(step)

			if self.drop is not None:
				out = self.drop(out)

			attns = []
			for _tmp, net in enumerate(self.nets):
				out, _attn, _state = net(inpute, states[_tmp], _src_pad_mask, None, out, True)
				states[_tmp] = _state
				attns.append(_attn)

			if self.out_normer is not None:
				out = self.out_normer(out)

			attns = torch.cat(attns, dim=1).permute(0, 2, 3, 1)
			_asize = attns.size()
			out = torch.cat([out, attns.contiguous().view(-1, _asize[-1]).mv(self.tattn_w.softmax(dim=0) if self.tattn_drop is None else self.tattn_drop(self.tattn_w).softmax(dim=0)).view(_asize[:-1]).bmm(inpute)], dim=-1)

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
				lpv = lpv.masked_fill(1 - done_trans.view(real_bsize, 1), ((step + 6.0) ** length_penalty) / lpv_base)

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

			_inds = (_inds / beam_size + torch.arange(0, real_bsize, beam_size, dtype=_inds.dtype, device=_inds.device).unsqueeze(1).expand_as(_inds)).view(real_bsize)

			# select the corresponding translation history for the top-k candidate and update translation records
			# trans: (bsize * beam_size, nquery) => (bsize * beam_size, nquery + 1)

			trans = torch.cat((trans.index_select(0, _inds), wds), 1)

			done_trans = (done_trans.view(real_bsize).index_select(0, _inds) + wds.eq(2).squeeze(1)).gt(0).view(bsize, beam_size)

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
			# states[i]: (bsize * beam_size, nquery, isize)
			# _inds: (bsize, beam_size) => (bsize * beam_size)

			for key, value in states.items():
				states[key] = value.index_select(0, _inds)

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

	def fix_init(self):

		if self.fbl is not None:
			_tbias = list(self.classifier.modules())[-1].bias
			for ind in self.fbl:
				_tbias.data[ind] = -1e32
