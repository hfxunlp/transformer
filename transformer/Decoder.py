#encoding: utf-8

import torch
from torch import nn
from modules.base import *
from utils.sampler import SampleMax
from utils.base import all_done, index_tensors, expand_bsize_for_beam, mask_tensor_type
from math import sqrt

from utils.fmt.base import pad_id

from cnfg.ihyp import *

class DecoderLayer(nn.Module):

	# isize: input size
	# fhsize: hidden size of PositionwiseFeedForward
	# attn_drop: dropout for MultiHeadAttention
	# num_head: number of heads in MultiHeadAttention
	# ahsize: hidden size of MultiHeadAttention
	# norm_residual: residue with layer normalized representation
	# k_rel_pos: window size (one side) of relative positional embeddings in self attention

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, ahsize=None, norm_residual=norm_residual_default, k_rel_pos=use_k_relative_position_decoder):

		super(DecoderLayer, self).__init__()

		_ahsize = isize if ahsize is None else ahsize
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		self.self_attn = SelfAttn(isize, _ahsize, isize, num_head=num_head, dropout=attn_drop, k_rel_pos=k_rel_pos, uni_direction_reduction=True)
		self.cross_attn = CrossAttn(isize, _ahsize, isize, num_head=num_head, dropout=attn_drop)

		self.ff = PositionwiseFF(isize, hsize=_fhsize, dropout=dropout, norm_residual=norm_residual)

		self.layer_normer1 = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)
		self.layer_normer2 = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

		self.drop = Dropout(dropout, inplace=True) if dropout > 0.0 else None

		self.norm_residual = norm_residual

	# inpute: encoded representation from encoder (bsize, seql, isize)
	# inputo: embedding of decoded translation (bsize, nquery, isize)
	# src_pad_mask: mask for given encoding source sentence (bsize, nquery, seql), see Encoder, expanded after generated with:
	#	src_pad_mask = input.eq(0).unsqueeze(1)
	# tgt_pad_mask: mask to hide the future input
	# query_unit: single query to decode, used to support decoding for given step

	def forward(self, inpute, inputo, src_pad_mask=None, tgt_pad_mask=None, query_unit=None):

		if query_unit is None:
			_inputo = self.layer_normer1(inputo)

			context = self.self_attn(_inputo, mask=tgt_pad_mask)

			if self.drop is not None:
				context = self.drop(context)

			context = context + (_inputo if self.norm_residual else inputo)

		else:
			_query_unit = self.layer_normer1(query_unit)

			context, states_return = self.self_attn(_query_unit, states=inputo)

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

class Decoder(nn.Module):

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
	# share_layer: using one shared decoder layer
	# disable_pemb: disable the standard positional embedding, can be enabled when use relative postional embeddings in self attention or AAN

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, emb_w=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindemb=True, forbidden_index=None, share_layer=False, disable_pemb=disable_std_pemb_decoder):

		super(Decoder, self).__init__()

		_ahsize = isize if ahsize is None else ahsize
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		self.drop = Dropout(dropout, inplace=True) if dropout > 0.0 else None

		self.xseql = xseql
		self.register_buffer('mask', torch.ones(xseql, xseql, dtype=mask_tensor_type).triu(1).unsqueeze(0))

		self.wemb = nn.Embedding(nwd, isize, padding_idx=pad_id)
		if emb_w is not None:
			self.wemb.weight = emb_w

		self.pemb = None if disable_pemb else PositionalEmb(isize, xseql, 0, 0)
		if share_layer:
			_shared_layer = DecoderLayer(isize, _fhsize, dropout, attn_drop, num_head, _ahsize)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([DecoderLayer(isize, _fhsize, dropout, attn_drop, num_head, _ahsize) for i in range(num_layer)])

		self.classifier = Linear(isize, nwd)
		# be careful since this line of code is trying to share the weight of the wemb and the classifier, which may cause problems if torch.nn updates
		if bindemb:
			self.classifier.weight = self.wemb.weight

		self.lsm = nn.LogSoftmax(-1)

		self.out_normer = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters) if norm_output else None

		self.fbl = None if forbidden_index is None else tuple(set(forbidden_index))

	# inpute: encoded representation from encoder (bsize, seql, isize)
	# inputo: decoded translation (bsize, nquery)
	# src_pad_mask: mask for given encoding source sentence (bsize, 1, seql), see Encoder, generated with:
	#	src_pad_mask = input.eq(0).unsqueeze(1)

	def forward(self, inpute, inputo, src_pad_mask=None):

		nquery = inputo.size(-1)

		out = self.wemb(inputo)

		out = out * sqrt(out.size(-1))
		if self.pemb is not None:
			out = out + self.pemb(inputo, expand=False)

		if self.drop is not None:
			out = self.drop(out)

		_mask = self._get_subsequent_mask(nquery)

		# the following line of code is to mask <pad> for the decoder,
		# which I think is useless, since only <pad> may pay attention to previous <pad> tokens, whos loss will be omitted by the loss function.
		#_mask = torch.gt(_mask + inputo.eq(0).unsqueeze(1), 0)

		for net in self.nets:
			out = net(inpute, out, src_pad_mask, _mask)

		if self.out_normer is not None:
			out = self.out_normer(out)

		out = self.lsm(self.classifier(out))

		return out

	def load_base(self, base_decoder):

		self.drop = base_decoder.drop

		self.wemb = base_decoder.wemb

		self.pemb = base_decoder.pemb

		_nets = list(base_decoder.nets)

		self.nets = nn.ModuleList(_nets[:len(self.nets)] + list(self.nets[len(_nets):]))

		self.classifier = base_decoder.classifier

		self.lsm = base_decoder.lsm

		self.out_normer = None if self.out_normer is None else base_decoder.out_normer

	def _get_subsequent_mask(self, length):

		return self.mask.narrow(1, 0, length).narrow(2, 0, length).contiguous() if length <= self.xseql else self.mask.new_ones(length, length).triu(1).unsqueeze(0)

	# this function repeats buffers of all cross-attention keys/values, corresponding inputs do not need to be repeated in beam search.

	def repeat_cross_attn_buffer(self, beam_size):

		for _m in self.modules():
			if isinstance(_m, (CrossAttn, MultiHeadAttn,)):
				_m.repeat_buffer(beam_size)

	# inpute: encoded representation from encoder (bsize, seql, isize)
	# src_pad_mask: mask for given encoding source sentence (bsize, seql), see Encoder, get by:
	#	src_pad_mask = input.eq(0).unsqueeze(1)
	# beam_size: the beam size for beam search
	# max_len: maximum length to generate

	def decode(self, inpute, src_pad_mask=None, beam_size=1, max_len=512, length_penalty=0.0, fill_pad=False):

		return self.beam_decode(inpute, src_pad_mask, beam_size, max_len, length_penalty, fill_pad=fill_pad) if beam_size > 1 else self.greedy_decode(inpute, src_pad_mask, max_len, fill_pad=fill_pad)

	# inpute: encoded representation from encoder (bsize, seql, isize)
	# src_pad_mask: mask for given encoding source sentence (bsize, 1, seql), see Encoder, generated with:
	#	src_pad_mask = input.eq(0).unsqueeze(1)
	# max_len: maximum length to generate
	# sample: for back translation

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
			out, _state = net(inpute, (None, None,), src_pad_mask, None, out)
			states[_tmp] = _state

		if self.out_normer is not None:
			out = self.out_normer(out)

		# out: (bsize, 1, nwd)
		# omit self.lsm for efficiency
		out = self.classifier(out)
		# wds: (bsize, 1)
		wds = SampleMax(out.softmax(-1), dim=-1, keepdim=False) if sample else out.argmax(dim=-1)

		trans = [wds]

		# done_trans: (bsize, 1)

		done_trans = wds.eq(2)

		for i in range(1, max_len):

			out = self.wemb(wds) * sqrt_isize
			if self.pemb is not None:
				out = out + self.pemb.get_pos(i)

			if self.drop is not None:
				out = self.drop(out)

			for _tmp, net in enumerate(self.nets):
				out, _state = net(inpute, states[_tmp], src_pad_mask, None, out)
				states[_tmp] = _state

			if self.out_normer is not None:
				out = self.out_normer(out)

			out = self.classifier(out)
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
			out, _state = net(inpute, (None, None,), src_pad_mask, None, out)
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

		# instead of update inpute: (bsize, seql, isize) => (bsize * beam_size, seql, isize) with the following line, we only update cross-attention buffers.
		#inpute = inpute.repeat(1, beam_size, 1).view(real_bsize, seql, isize)

		self.repeat_cross_attn_buffer(beam_size)

		# _src_pad_mask: (bsize, 1, seql) => (bsize * beam_size, 1, seql)

		_src_pad_mask = None if src_pad_mask is None else src_pad_mask.repeat(1, beam_size, 1).view(real_bsize, 1, seql)

		# states[i]: (bsize, 1, isize) => (bsize * beam_size, 1, isize)

		states = expand_bsize_for_beam(states, beam_size=beam_size)

		for step in range(1, max_len):

			out = self.wemb(wds) * sqrt_isize
			if self.pemb is not None:
				out = out + self.pemb.get_pos(step)

			if self.drop is not None:
				out = self.drop(out)

			for _tmp, net in enumerate(self.nets):
				out, _state = net(inpute, states[_tmp], _src_pad_mask, None, out)
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
				lpv.masked_fill_(~done_trans.view(real_bsize, 1), ((step + 6.0) ** length_penalty) / lpv_base)

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

	# inpute: encoded representation from encoder (bsize, seql, isize)

	def get_sos_emb(self, inpute, bsize=None):

		bsize = inpute.size(0) if bsize is None else bsize

		return self.wemb.weight[1].view(1, 1, -1).expand(bsize, 1, -1)

	def fix_init(self):

		self.fix_load()
		with torch.no_grad():
			#self.wemb.weight[pad_id].zero_()
			self.classifier.weight[pad_id].zero_()

	def fix_load(self):

		if self.fbl is not None:
			with torch.no_grad():
				self.classifier.bias.index_fill_(0, torch.tensor(self.fbl, dtype=torch.long, device=self.classifier.bias.device), -inf_default)

	def unbind_classifier_weight(self):

		if self.classifier.weight.is_set_to(self.wemb.weight):
			_tmp = self.classifier.weight
			_new_w = nn.Parameter(torch.Tensor(_tmp.size()))
			with torch.no_grad():
				_new_w.data.copy_(_tmp.data)
			self.classifier.weight = _new_w

	# this function will untie the decoder embedding from the encoder

	def update_vocab(self, indices):

		_nwd = len(indices)
		_wemb = nn.Embedding(_nwd, self.wemb.weight.size(-1), padding_idx=pad_id)
		_classifier = Linear(self.classifier.weight.size(-1), _nwd)
		with torch.no_grad():
			_wemb.weight.copy_(self.wemb.weight.index_select(0, indices))
			if self.classifier.weight.is_set_to(self.wemb.weight):
				_classifier.weight = _wemb.weight
			else:
				_classifier.weight.copy_(self.classifier.weight.index_select(0, indices))
			_classifier.bias.copy_(self.classifier.bias.index_select(0, indices))
		self.wemb, self.classifier = _wemb, _classifier

	def index_cross_attn_buffer(self, indices, dim=0):

		for _m in self.modules():
			if isinstance(_m, (CrossAttn, MultiHeadAttn,)):
				_m.index_buffer(indices, dim=dim)

	# inpute: encoded representation from encoder (bsize, seql, isize)
	# src_pad_mask: mask for given encoding source sentence (bsize, seql), see Encoder, get by:
	#	src_pad_mask = input.eq(0).unsqueeze(1)
	# beam_size: the beam size for beam search
	# max_len: maximum length to generate

	def decode_clip(self, inpute, src_pad_mask, beam_size=1, max_len=512, length_penalty=0.0, return_mat=True):

		return self.beam_decode_clip(inpute, src_pad_mask, beam_size, max_len, length_penalty, return_mat) if beam_size > 1 else self.greedy_decode_clip(inpute, src_pad_mask, max_len, return_mat)

	# inpute: encoded representation from encoder (bsize, seql, isize)
	# src_pad_mask: mask for given encoding source sentence (bsize, 1, seql), see Encoder, generated with:
	#	src_pad_mask = input.eq(0).unsqueeze(1)
	# max_len: maximum length to generate

	def greedy_decode_clip(self, inpute, src_pad_mask=None, max_len=512, return_mat=True):

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
			out, _state = net(inpute, (None, None,), src_pad_mask, None, out)
			states[_tmp] = _state

		if self.out_normer is not None:
			out = self.out_normer(out)

		# out: (bsize, 1, nwd)

		out = self.lsm(self.classifier(out))

		# wds: (bsize, 1)

		wds = out.argmax(dim=-1)

		mapper = list(range(bsize))
		rs = [None for i in range(bsize)]

		trans = [wds]

		for i in range(1, max_len):

			out = self.wemb(wds) * sqrt_isize
			if self.pemb is not None:
				out = out + self.pemb.get_pos(i)

			if self.drop is not None:
				out = self.drop(out)

			for _tmp, net in enumerate(self.nets):
				out, _state = net(inpute, states[_tmp], src_pad_mask, None, out)
				states[_tmp] = _state

			if self.out_normer is not None:
				out = self.out_normer(out)

			# out: (bsize, 1, nwd)
			out = self.lsm(self.classifier(out))
			wds = out.argmax(dim=-1)

			trans.append(wds)

			# done_trans: (bsize)
			done_trans = wds.squeeze(1).eq(2)

			_ndone = done_trans.int().sum().item()
			if _ndone == bsize:
				for _iu, _tran in enumerate(torch.cat(trans, 1).unbind(0)):
					rs[mapper[_iu]] = _tran
				break
			elif _ndone > 0:
				_dind = done_trans.nonzero().squeeze(1)
				_trans = torch.cat(trans, 1)
				for _iu, _tran in zip(_dind.tolist(), _trans.index_select(0, _dind).unbind(0)):
					rs[mapper[_iu]] = _tran

				# reduce bsize for not finished decoding
				_ndid = (~done_trans).nonzero().squeeze(1)
				bsize = _ndid.size(0)
				wds = wds.index_select(0, _ndid)
				#inpute = inpute.index_select(0, _ndid)
				self.index_cross_attn_buffer(_ndid)
				if src_pad_mask is not None:
					src_pad_mask = src_pad_mask.index_select(0, _ndid)
				states = index_tensors(states, indices=_ndid, dim=0)
				trans = list(_trans.index_select(0, _ndid).unbind(1))

				# update mapper
				for _ind, _iu in enumerate(_ndid.tolist()):
					mapper[_ind] = mapper[_iu]

		return torch.stack(pad_tensors(rs), 0) if return_mat else rs

	# inpute: encoded representation from encoder (bsize, seql, isize)
	# src_pad_mask: mask for given encoding source sentence (bsize, 1, seql), see Encoder, generated with:
	#	src_pad_mask = input.eq(0).unsqueeze(1)
	# beam_size: beam size
	# max_len: maximum length to generate

	def beam_decode_clip(self, inpute, src_pad_mask=None, beam_size=8, max_len=512, length_penalty=0.0, return_mat=True, return_all=False, clip_beam=clip_beam_with_lp):

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
			out, _state = net(inpute, (None, None,), src_pad_mask, None, out)
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
		#inpute = inpute.repeat(1, beam_size, 1).view(real_bsize, seql, isize)

		self.repeat_cross_attn_buffer(beam_size)

		# _src_pad_mask: (bsize, 1, seql) => (bsize * beam_size, 1, seql)

		_src_pad_mask = None if src_pad_mask is None else src_pad_mask.repeat(1, beam_size, 1).view(real_bsize, 1, seql)

		# states[i]: (bsize, 1, isize) => (bsize * beam_size, 1, isize)

		states = expand_bsize_for_beam(states, beam_size=beam_size)

		mapper = list(range(bsize))
		rs = [None for i in range(bsize)]
		if return_all:
			rscore = [None for i in range(bsize)]

		for step in range(1, max_len):

			out = self.wemb(wds) * sqrt_isize
			if self.pemb is not None:
				out = out + self.pemb.get_pos(step)

			if self.drop is not None:
				out = self.drop(out)

			for _tmp, net in enumerate(self.nets):
				out, _state = net(inpute, states[_tmp], _src_pad_mask, None, out)
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
			# mask_from_done_trans_u: (bsize, beam_size) => (bsize, beam_size * beam_size)
			# added_scores: (bsize, 1, beam_size) => (bsize, beam_size, beam_size)

			_scores, _wds = out.topk(beam_size, dim=-1)
			_scores = (_scores.masked_fill(done_trans.unsqueeze(2).expand(bsize, beam_size, beam_size), 0.0) + sum_scores.unsqueeze(2).expand(bsize, beam_size, beam_size))

			if length_penalty > 0.0:
				lpv.masked_fill_(~done_trans.view(real_bsize, 1), ((step + 6.0) ** length_penalty) / lpv_base)

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

			# using "_inds / beam_size" in case old pytorch does not support "//" operation
			_inds = (_inds // beam_size + torch.arange(0, real_bsize, beam_size, dtype=_inds.dtype, device=_inds.device).unsqueeze(1).expand_as(_inds)).view(real_bsize)

			# select the corresponding translation history for the top-k candidate and update translation records
			# trans: (bsize * beam_size, nquery) => (bsize * beam_size, nquery + 1)

			trans = torch.cat((trans.index_select(0, _inds), wds), 1)

			done_trans = (done_trans.view(real_bsize).index_select(0, _inds) | wds.eq(2).squeeze(1)).view(bsize, beam_size)

			# check early stop for beam search
			# done_trans: (bsize, beam_size)
			# scores: (bsize, beam_size)

			if length_penalty > 0.0:
				lpv = lpv.index_select(0, _inds)
				_done_trans_u = done_trans.sum(1).eq(beam_size)
			elif return_all:
				_done_trans_u = done_trans.sum(1).eq(beam_size)
			else:
				_done_trans_u = done_trans.select(1, 0)

			# check beam states(done or not)

			_ndone = _done_trans_u.int().sum().item()
			if _ndone == bsize:
				if (not clip_beam) and (length_penalty > 0.0):
					scores = scores / lpv.view(bsize, beam_size)
					scores, _inds = scores.topk(beam_size, dim=-1)
					_inds = (_inds + torch.arange(0, real_bsize, beam_size, dtype=_inds.dtype, device=_inds.device).unsqueeze(1).expand_as(_inds)).view(real_bsize)
					trans = trans.view(real_bsize, -1).index_select(0, _inds).view(bsize, beam_size, -1)
				if return_all:
					for _iu, (_tran, _score) in enumerate(zip(trans.view(bsize, beam_size, -1).unbind(0), scores.view(bsize, beam_size).unbind(0))):
						_rid = mapper[_iu]
						rs[_rid] = _tran
						rscore[_rid] = _score
				else:
					for _iu, _tran in enumerate(trans.view(bsize, beam_size, -1).unbind(0)):
						rs[mapper[_iu]] = _tran[0]
				break

			# update the corresponding hidden states
			# states[i]: (bsize * beam_size, nquery, isize)
			# _inds: (bsize, beam_size) => (bsize * beam_size)

			states = index_tensors(states, indices=_inds, dim=0)

			if _ndone > 0:
				_dind = _done_trans_u.nonzero().squeeze(1)
				_trans = trans.view(bsize, beam_size, -1)
				_trans_sel = _trans.index_select(0, _dind)

				if (not clip_beam) and (length_penalty > 0.0):
					_scores_sel = scores.index_select(0, _dind) / lpv.view(bsize, beam_size).index_select(0, _dind)
					_sel_bsize = _dind.size(0)
					_sel_real_bsize = _sel_bsize * beam_size
					_scores_sel, _inds = _scores_sel.topk(beam_size, dim=-1)
					_inds = (_inds + torch.arange(0, _sel_real_bsize, beam_size, dtype=_inds.dtype, device=_inds.device).unsqueeze(1).expand_as(_inds)).view(_sel_real_bsize)
					_trans_sel = _trans_sel.view(_sel_real_bsize, -1).index_select(0, _inds).view(_sel_bsize, beam_size, -1)
				if return_all:
						for _iu, _tran, _score in zip(_dind.tolist(), _trans_sel.unbind(0), _scores_sel.unbind(0)):
							_rid = mapper[_iu]
							rs[_rid] = _tran
							rscore[_rid] = _score
				else:
						for _iu, _tran in zip(_dind.tolist(), _trans_sel.unbind(0)):
							rs[mapper[_iu]] = _tran[0]

				# reduce bsize for not finished decoding
				_ndid = (~_done_trans_u).nonzero().squeeze(1)

				_bsize = _ndid.size(0)
				bsizeb2 = _bsize * beam_size2
				_real_bsize = _bsize * beam_size

				wds = wds.view(bsize, beam_size).index_select(0, _ndid).view(_real_bsize, 1)
				#inpute = inpute.view(bsize, beam_size, seql, isize).index_select(0, _ndid).view(_real_bsize, seql, isize)
				for _m in self.modules():
					if isinstance(layer, (CrossAttn, MultiHeadAttn,)) and layer.real_iK is not None:
						layer.real_iK, layer.real_iV = tuple(_vu.view(bsize, beam_size, *list(_vu.size()[1:])).index_select(0, _ndid).view(_real_bsize, *list(_vu.size()[1:])) for _vu in (layer.real_iK, layer.real_iV,))
				if _src_pad_mask is not None:
					_src_pad_mask = _src_pad_mask.view(bsize, beam_size, 1, seql).index_select(0, _ndid).view(_real_bsize, 1, seql)
				for k, value in states.items():
					states[k] = [_vu.view(bsize, beam_size, *list(_vu.size()[1:])).index_select(0, _ndid).view(_real_bsize, *list(_vu.size()[1:])) for _vu in value]
				sum_scores = sum_scores.index_select(0, _ndid)
				trans = _trans.index_select(0, _ndid).view(_real_bsize, -1)
				if length_penalty > 0.0:
					lpv = lpv.view(bsize, beam_size).index_select(0, _ndid).view(_real_bsize, 1)
				done_trans = done_trans.index_select(0, _ndid)

				bsize, real_bsize = _bsize, _real_bsize

				# update mapper
				for _ind, _iu in enumerate(_ndid.tolist()):
					mapper[_ind] = mapper[_iu]

		if return_mat:
			rs = torch.stack(pad_tensors(rs), 0)

		if return_all:

			return rs, torch.stack(rscore, 0)
		else:

			return rs
