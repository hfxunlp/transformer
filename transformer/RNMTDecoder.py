#encoding: utf-8

# Difference: RNMT decoder concat attention and decoder layer output and directly classify with it, which makes the sharing parameters between classifier and embedding impossible, this implementation optionally reduce the concatenated dimension with another Linear transform followed by tanh like GlobalAttention

import torch
from torch import nn

from modules.base import CrossAttn, Dropout, Linear
from modules.rnncells import LSTMCell4RNMT, prepare_initState
from transformer.Decoder import Decoder as DecoderBase
from utils.fmt.parser import parse_none
from utils.sampler import SampleMax
from utils.torch.comp import all_done#, torch_no_grad

from cnfg.ihyp import *
from cnfg.vocab.base import eos_id, pad_id

class FirstLayer(nn.Module):

	# isize: input size
	# osize: output size
	def __init__(self, isize, osize=None, dropout=0.0, **kwargs):

		super(FirstLayer, self).__init__()

		_osize = parse_none(osize, isize)

		self.net = LSTMCell4RNMT(isize, _osize)
		self.init_hx = nn.Parameter(torch.zeros(1, _osize))
		self.init_cx = nn.Parameter(torch.zeros(1, _osize))

		self.drop = Dropout(dropout, inplace=False) if dropout > 0.0 else None

	# inputo: embedding of decoded translation (bsize, nquery, isize)
	# query_unit: single query to decode, used to support decoding for given step

	def forward(self, inputo, states=None, first_step=False, **kwargs):

		if states is None:
			hx, cx = prepare_initState(self.init_hx, self.init_cx, inputo.size(0))
			outs = []

			for _du in inputo.unbind(1):
				hx, cx = self.net(_du, (hx, cx))
				outs.append(hx)

			outs = torch.stack(outs, 1)

			if self.drop is not None:
				outs = self.drop(outs)

			return outs
		else:
			hx, cx = self.net(inputo, prepare_initState(self.init_hx, self.init_cx, inputo.size(0)) if first_step else states)

			out = hx if self.drop is None else self.drop(hx)

			return out, (hx, cx)

class DecoderLayer(nn.Module):

	# isize: input size
	# osize: output size
	def __init__(self, isize, osize=None, dropout=0.0, residual=True, **kwargs):

		super(DecoderLayer, self).__init__()

		_osize = parse_none(osize, isize)

		self.net = LSTMCell4RNMT(isize + _osize, _osize)
		self.init_hx = nn.Parameter(torch.zeros(1, _osize))
		self.init_cx = nn.Parameter(torch.zeros(1, _osize))

		self.drop = Dropout(dropout, inplace=False) if dropout > 0.0 else None

		self.residual = residual

	# inputo: embedding of decoded translation (bsize, nquery, isize)
	# query_unit: single query to decode, used to support decoding for given step

	def forward(self, inputo, attn, states=None, first_step=False, **kwargs):

		if states is None:
			hx, cx = prepare_initState(self.init_hx, self.init_cx, inputo.size(0))
			outs = []

			_inputo = torch.cat((inputo, attn), -1)

			for _du in _inputo.unbind(1):
				hx, cx = self.net(_du, (hx, cx))
				outs.append(hx)

			outs = torch.stack(outs, 1)

			if self.drop is not None:
				outs = self.drop(outs)

			return outs + inputo if self.residual else outs
		else:

			hx, cx = self.net(torch.cat((inputo, attn), -1), prepare_initState(self.init_hx, self.init_cx, inputo.size(0)) if first_step else states)

			out = hx if self.drop is None else self.drop(hx)

			return out + inputo if self.residual else out, (hx, cx)

class Decoder(DecoderBase):

	# isize: size of word embedding
	# nwd: number of words
	# num_layer: number of layers
	# attn_drop: dropout for MultiHeadAttention
	# emb_w: weight for embedding. Use only when the encoder and decoder share a same dictionary
	# num_head: number of heads in MultiHeadAttention
	# xseql: maxmimum length of sequence
	# ahsize: number of hidden units for MultiHeadAttention
	# bindemb: bind embedding and classifier weight

	def __init__(self, isize, nwd, num_layer, dropout=0.0, attn_drop=0.0, act_drop=None, emb_w=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindemb=False, forbidden_index=None, projector=True, **kwargs):

		_ahsize = parse_none(ahsize, isize)

		super(Decoder, self).__init__(isize, nwd, num_layer, fhsize=isize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindemb=bindemb, forbidden_index=forbidden_index, **kwargs)

		self.flayer = FirstLayer(isize, osize=isize, dropout=dropout)

		self.attn = CrossAttn(isize, _ahsize, isize, num_head, dropout=attn_drop)

		self.nets = nn.ModuleList([DecoderLayer(isize, isize, dropout, i > 0) for i in range(num_layer - 1)])

		self.projector = Linear(isize, isize, bias=False) if projector else None

		self.classifier = Linear(isize * 2, nwd)#nn.Sequential(Linear(isize * 2, isize, bias=False), nn.Tanh(), Linear(isize, nwd))
		# be careful since this line of code is trying to share the weight of the wemb and the classifier, which may cause problems if torch.nn updates
		#if bindemb:
			#list(self.classifier.modules())[-1].weight = self.wemb.weight

		self.mask = None

	# inpute: encoded representation from encoder (bsize, seql, isize)
	# inputo: decoded translation (bsize, nquery)
	# src_pad_mask: mask for given encoding source sentence (bsize, 1, seql), see Encoder, generated with:
	#	src_pad_mask = input.eq(pad_id).unsqueeze(1)

	def forward(self, inpute, inputo, src_pad_mask=None, **kwargs):

		out = self.wemb(inputo)

		if self.drop is not None:
			out = self.drop(out)

		out = self.flayer(out)

		if self.projector:
			inpute = self.projector(inpute)

		attn = self.attn(out, inpute, src_pad_mask)

		# the following line of code is to mask <pad> for the decoder,
		# which I think is useless, since only <pad> may pay attention to previous <pad> tokens, whos loss will be omitted by the loss function.
		#_mask = torch.gt(_mask + inputo.eq(pad_id).unsqueeze(1), 0)

		for net in self.nets:
			out = net(out, attn)

		if self.out_normer is not None:
			out = self.out_normer(out)

		out = self.lsm(self.classifier(torch.cat((out, attn), -1)))

		return out

	# inpute: encoded representation from encoder (bsize, seql, isize)
	# src_pad_mask: mask for given encoding source sentence (bsize, 1, seql), see Encoder, generated with:
	#	src_pad_mask = input.eq(pad_id).unsqueeze(1)
	# max_len: maximum length to generate

	def greedy_decode(self, inpute, src_pad_mask=None, max_len=512, fill_pad=False, sample=False, **kwargs):

		bsize = inpute.size(0)

		out = self.get_sos_emb(inpute)

		# out: input to the decoder for the first step (bsize, 1, isize)

		if self.drop is not None:
			out = self.drop(out)

		out, statefl = self.flayer(out, "init", True)

		states = {}

		if self.projector:
			inpute = self.projector(inpute)

		attn = self.attn(out.unsqueeze(1), inpute, src_pad_mask).squeeze(1)

		for _tmp, net in enumerate(self.nets):
			out, _state = net(out, attn, "init", True)
			states[_tmp] = _state

		if self.out_normer is not None:
			out = self.out_normer(out)

		# out: (bsize, nwd)
		out = self.classifier(torch.cat((out, attn), -1))
		# wds: (bsize)
		wds = SampleMax(out.softmax(-1), dim=-1, keepdim=False) if sample else out.argmax(dim=-1)

		trans = [wds]

		# done_trans: (bsize)

		done_trans = wds.eq(eos_id)

		for i in range(1, max_len):

			out = self.wemb(wds)

			if self.drop is not None:
				out = self.drop(out)

			out, statefl = self.flayer(out, statefl)

			attn = self.attn(out.unsqueeze(1), inpute, src_pad_mask).squeeze(1)

			for _tmp, net in enumerate(self.nets):
				out, _state = net(out, attn, states[_tmp])
				states[_tmp] = _state

			if self.out_normer is not None:
				out = self.out_normer(out)

			out = self.classifier(torch.cat((out, attn), -1))
			wds = SampleMax(out.softmax(-1), dim=-1, keepdim=False) if sample else out.argmax(dim=-1)

			trans.append(wds.masked_fill(done_trans, pad_id) if fill_pad else wds)

			done_trans = done_trans | wds.eq(eos_id)
			if all_done(done_trans, bsize):
				break

		return torch.stack(trans, 1)

	# inpute: encoded representation from encoder (bsize, seql, isize)
	# src_pad_mask: mask for given encoding source sentence (bsize, 1, seql), see Encoder, generated with:
	#	src_pad_mask = input.eq(pad_id).unsqueeze(1)
	# beam_size: beam size
	# max_len: maximum length to generate

	def beam_decode(self, inpute, src_pad_mask=None, beam_size=8, max_len=512, length_penalty=0.0, return_all=False, clip_beam=clip_beam_with_lp, fill_pad=False, **kwargs):

		bsize, seql = inpute.size()[:2]

		beam_size2 = beam_size * beam_size
		bsizeb2 = bsize * beam_size2
		real_bsize = bsize * beam_size

		out = self.get_sos_emb(inpute)

		if length_penalty > 0.0:
			# lpv: length penalty vector for each beam (bsize * beam_size, 1)
			lpv = out.new_ones(real_bsize, 1)
			lpv_base = 6.0 ** length_penalty

		if self.drop is not None:
			out = self.drop(out)

		out, statefl = self.flayer(out, "init", True)
		statefl = torch.stack(statefl, -2)

		states = {}

		if self.projector:
			inpute = self.projector(inpute)

		attn = self.attn(out.unsqueeze(1), inpute, src_pad_mask).squeeze(1)

		for _tmp, net in enumerate(self.nets):
			out, _state = net(out, attn, "init", True)
			states[_tmp] = torch.stack(_state, -2)

		if self.out_normer is not None:
			out = self.out_normer(out)

		# out: (bsize, nwd)

		out = self.lsm(self.classifier(torch.cat((out, attn), -1)))

		# scores: (bsize, beam_size) => (bsize, beam_size)
		# wds: (bsize * beam_size)
		# trans: (bsize * beam_size, 1)

		scores, wds = out.topk(beam_size, dim=-1)
		sum_scores = scores
		wds = wds.view(real_bsize)
		trans = wds.unsqueeze(1)
		_inds_add_beam2 = torch.arange(0, bsizeb2, beam_size2, dtype=wds.dtype, device=wds.device).unsqueeze(1).expand(bsize, beam_size)
		_inds_add_beam = torch.arange(0, real_bsize, beam_size, dtype=wds.dtype, device=wds.device).unsqueeze(1).expand(bsize, beam_size)

		# done_trans: (bsize, beam_size)

		done_trans = wds.view(bsize, beam_size).eq(eos_id)

		# inpute: (bsize, seql, isize) => (bsize * beam_size, seql, isize)

		self.repeat_cross_attn_buffer(beam_size)

		# _src_pad_mask: (bsize, 1, seql) => (bsize * beam_size, 1, seql)

		_src_pad_mask = None if src_pad_mask is None else src_pad_mask.repeat(1, beam_size, 1).view(real_bsize, 1, seql)

		# states[i]: (bsize, 2, isize) => (bsize * beam_size, 2, isize)

		statefl = statefl.repeat(1, beam_size, 1).view(real_bsize, 2, isize)
		states = expand_bsize_for_beam(states, beam_size=beam_size)

		for step in range(1, max_len):

			out = self.wemb(wds)

			if self.drop is not None:
				out = self.drop(out)

			out, statefl = self.flayer(out, statefl.unbind(-2))
			statefl = torch.stack(statefl, -2)

			attn = self.attn(out.unsqueeze(1), inpute, _src_pad_mask).squeeze(1)

			for _tmp, net in enumerate(self.nets):
				out, _state = net(out, attn, states[_tmp].unbind(-2))
				states[_tmp] = torch.stack(_state, -2)

			if self.out_normer is not None:
				out = self.out_normer(out)

			# out: (bsize, beam_size, nwd)

			out = self.lsm(self.classifier(torch.cat((out, attn), -1))).view(bsize, beam_size, -1)

			# find the top k ** 2 candidates and calculate route scores for them
			# _scores: (bsize, beam_size, beam_size)
			# done_trans: (bsize, beam_size)
			# scores: (bsize, beam_size)
			# _wds: (bsize, beam_size, beam_size)
			# mask_from_done_trans: (bsize, beam_size) => (bsize, beam_size * beam_size)
			# added_scores: (bsize, 1, beam_size) => (bsize, beam_size, beam_size)

			_scores, _wds = out.topk(beam_size, dim=-1)
			_done_trans_unsqueeze = done_trans.unsqueeze(2)
			_scores = (_scores.masked_fill(_done_trans_unsqueeze.expand(bsize, beam_size, beam_size), 0.0) + sum_scores.unsqueeze(2).repeat(1, 1, beam_size).masked_fill_(select_zero_(_done_trans_unsqueeze.repeat(1, 1, beam_size), -1, 0), -inf_default))

			if length_penalty > 0.0:
				lpv.masked_fill_(~done_trans.view(real_bsize, 1), ((step + 6.0) ** length_penalty) / lpv_base)

			# clip from k ** 2 candidate and remain the top-k for each path
			# scores: (bsize, beam_size * beam_size) => (bsize, beam_size)
			# _inds: indexes for the top-k candidate (bsize, beam_size)

			if clip_beam and (length_penalty > 0.0):
				scores, _inds = (_scores.view(real_bsize, beam_size) / lpv.expand(real_bsize, beam_size)).view(bsize, beam_size2).topk(beam_size, dim=-1)
				_tinds = (_inds + _inds_add_beam2).view(real_bsize)
				sum_scores = _scores.view(bsizeb2).index_select(0, _tinds).view(bsize, beam_size)
			else:
				scores, _inds = _scores.view(bsize, beam_size2).topk(beam_size, dim=-1)
				_tinds = (_inds + _inds_add_beam2).view(real_bsize)
				sum_scores = scores

			# select the top-k candidate with higher route score and update translation record
			# wds: (bsize, beam_size, beam_size) => (bsize * beam_size)

			wds = _wds.view(bsizeb2).index_select(0, _tinds)

			# reduces indexes in _inds from (beam_size ** 2) to beam_size
			# thus the fore path of the top-k candidate is pointed out
			# _inds: indexes for the top-k candidate (bsize, beam_size)

			_inds = (_inds // beam_size + _inds_add_beam).view(real_bsize)

			# select the corresponding translation history for the top-k candidate and update translation records
			# trans: (bsize * beam_size, nquery) => (bsize * beam_size, nquery + 1)

			trans = torch.cat((trans.index_select(0, _inds), (wds.masked_fill(done_trans.view(real_bsize), pad_id) if fill_pad else wds).unsqueeze(1)), 1)

			done_trans = (done_trans.view(real_bsize).index_select(0, _inds) & wds.eq(eos_id)).view(bsize, beam_size)

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
			# states[i]: (bsize * beam_size, 2, isize)
			# _inds: (bsize, beam_size) => (bsize * beam_size)

			statefl = statefl.index_select(0, _inds)
			states = index_tensors(states, indices=_inds, dim=0)

		# if length penalty is only applied in the last step, apply length penalty
		if (not clip_beam) and (length_penalty > 0.0):
			scores = scores / lpv.view(bsize, beam_size)
			scores, _inds = scores.topk(beam_size, dim=-1)
			_inds = (_inds + _inds_add_beam).view(real_bsize)
			trans = trans.view(real_bsize, -1).index_select(0, _inds)

		if return_all:

			return trans.view(bsize, beam_size, -1), scores
		else:

			return trans.view(bsize, beam_size, -1).select(1, 0)

	"""def fix_load(self):

		if self.fbl is not None:
			with torch_no_grad():
				list(self.classifier.modules())[-1].bias.index_fill_(0, torch.as_tensor(self.fbl, dtype=torch.long, device=self.classifier.bias.device), -inf_default)"""
