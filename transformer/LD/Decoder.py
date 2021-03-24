#encoding: utf-8

import torch
from torch import nn

from modules.base import CrossAttn, ResidueCombiner
from utils.sampler import SampleMax
from modules.TA import PositionwiseFF

from utils.base import all_done, index_tensors, expand_bsize_for_beam, repeat_bsize_for_beam_tensor
from math import sqrt

from utils.fmt.base import pad_id

from transformer.Decoder import DecoderLayer as DecoderLayerBase
from transformer.Decoder import Decoder as DecoderBase

from cnfg.ihyp import *

class DecoderLayer(DecoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, ahsize=None, **kwargs):

		_ahsize = isize if ahsize is None else ahsize

		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(DecoderLayer, self).__init__(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, ahsize=_ahsize, **kwargs)

		self.cattn = CrossAttn(isize, _ahsize, isize, num_head, dropout=attn_drop)

		self.ff = PositionwiseFF(isize, _fhsize, dropout)
		self.scff = ResidueCombiner(isize, 2, _fhsize, dropout)

	def forward(self, inpute, inputh, inputo, src_pad_mask=None, chk_pad_mask=None, tgt_pad_mask=None, query_unit=None):

		if query_unit is None:

			context = self.self_attn(inputo, mask=tgt_pad_mask)

			if self.drop is not None:
				context = self.drop(context)

			context = context + inputo

		else:

			context, states_return = self.self_attn(query_unit, states=inputo)

			if self.drop is not None:
				context = self.drop(context)

			context = context + query_unit

		_context = self.layer_normer1(context)

		_context = self.scff(_context, self.cattn(_context, inputh, mask=chk_pad_mask))

		_context_new = self.cross_attn(_context, inpute, mask=src_pad_mask)

		if self.drop is not None:
			_context_new = self.drop(_context_new)

		context = self.layer_normer2(_context_new + _context)

		context = self.ff(context)

		if query_unit is None:
			return context
		else:
			return context, states_return

class Decoder(DecoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, emb_w=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindemb=False, forbidden_index=None, share_layer=False, **kwargs):

		_ahsize = isize if ahsize is None else ahsize

		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Decoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindemb=bindemb, forbidden_index=forbidden_index, share_layer=share_layer, **kwargs)

		if share_layer:
			_shared_layer = DecoderLayer(isize, _fhsize, dropout, attn_drop, num_head, _ahsize)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([DecoderLayer(isize, _fhsize, dropout, attn_drop, num_head, _ahsize) for i in range(num_layer)])

	def forward(self, inpute, inputh, inputo, src_pad_mask=None, chk_pad_mask=None):

		bsize, nquery = inputo.size()

		out = self.wemb(inputo)

		out = out * sqrt(out.size(-1))
		if self.pemb is not None:
			out = out + self.pemb(inputo, expand=False)

		if self.drop is not None:
			out = self.drop(out)

		out = self.out_normer(out)

		_mask = self._get_subsequent_mask(nquery)

		for net, inputu, inputhu in zip(self.nets, inpute.unbind(dim=-1), inputh.unbind(dim=-1)):
			out = net(inputu, inputhu, out, src_pad_mask, chk_pad_mask, _mask)

		out = self.lsm(self.classifier(out))

		return out

	def greedy_decode(self, inpute, inputh, src_pad_mask=None, chk_pad_mask=None, max_len=512, fill_pad=False, sample=False):

		bsize = inpute.size(0)

		sos_emb = self.get_sos_emb(inpute)

		sqrt_isize = sqrt(sos_emb.size(-1))

		out = sos_emb * sqrt_isize
		if self.pemb is not None:
			 out = out + self.pemb.get_pos(0)

		if self.drop is not None:
			out = self.drop(out)

		out = self.out_normer(out)

		states = {}

		for _tmp, (net, inputu, inputhu) in enumerate(zip(self.nets, inpute.unbind(dim=-1), inputh.unbind(dim=-1))):
			out, _state = net(inputu, inputhu, None, src_pad_mask, chk_pad_mask, None, out, True)
			states[_tmp] = _state

		out = self.classifier(out)
		wds = SampleMax(out.softmax(-1), dim=-1, keepdim=False) if sample else out.argmax(dim=-1)

		trans = [wds]

		done_trans = wds.eq(2)

		for i in range(1, max_len):

			out = self.wemb(wds) * sqrt_isize
			if self.pemb is not None:
				out = out + self.pemb.get_pos(i)

			if self.drop is not None:
				out = self.drop(out)

			out = self.out_normer(out)

			for _tmp, (net, inputu, inputhu) in enumerate(zip(self.nets, inpute.unbind(dim=-1), inputh.unbind(dim=-1))):
				out, _state = net(inputu, inputhu, states[_tmp], src_pad_mask, chk_pad_mask, None, out, True)
				states[_tmp] = _state

			out = self.classifier(out)
			wds = SampleMax(out.softmax(-1), dim=-1, keepdim=False) if sample else out.argmax(dim=-1)

			trans.append(wds.masked_fill(done_trans, pad_id) if fill_pad else wds)

			done_trans = done_trans | wds.eq(2)
			if all_done(done_trans, bsize):
				break

		return torch.cat(trans, 1)

	def beam_decode(self, inpute, inputh, src_pad_mask=None, chk_pad_mask=None, beam_size=8, max_len=512, length_penalty=0.0, return_all=False, clip_beam=clip_beam_with_lp, fill_pad=False):

		bsize, seql = inpute.size()[:2]

		beam_size2 = beam_size * beam_size
		bsizeb2 = bsize * beam_size2
		real_bsize = bsize * beam_size

		sos_emb = self.get_sos_emb(inpute)
		isize = sos_emb.size(-1)
		sqrt_isize = sqrt(isize)

		if length_penalty > 0.0:
			lpv = sos_emb.new_ones(real_bsize, 1)
			lpv_base = 6.0 ** length_penalty

		out = sos_emb * sqrt_isize
		if self.pemb is not None:
			 out = out + self.pemb.get_pos(0)

		if self.drop is not None:
			out = self.drop(out)

		out = self.out_normer(out)

		states = {}

		for _tmp, (net, inputu, inputhu) in enumerate(zip(self.nets, inpute.unbind(dim=-1), inputh.unbind(dim=-1))):
			out, _state = net(inputu, inputhu, None, src_pad_mask, chk_pad_mask, None, out, True)
			states[_tmp] = _state

		out = self.lsm(self.classifier(out))

		scores, wds = out.topk(beam_size, dim=-1)
		scores = scores.squeeze(1)
		sum_scores = scores
		wds = wds.view(real_bsize, 1)
		trans = wds

		done_trans = wds.view(bsize, beam_size).eq(2)

		#inputh = repeat_bsize_for_beam_tensor(inputh, beam_size)
		self.repeat_cross_attn_buffer(beam_size)

		_src_pad_mask = None if src_pad_mask is None else src_pad_mask.repeat(1, beam_size, 1).view(real_bsize, 1, seql)
		_chk_pad_mask = None if chk_pad_mask is None else repeat_bsize_for_beam_tensor(chk_pad_mask, beam_size)

		states = expand_bsize_for_beam(states, beam_size=beam_size)

		for step in range(1, max_len):

			out = self.wemb(wds) * sqrt_isize
			if self.pemb is not None:
				out = out + self.pemb.get_pos(step)

			if self.drop is not None:
				out = self.drop(out)

			out = self.out_normer(out)

			for _tmp, (net, inputu, inputhu) in enumerate(zip(self.nets, inpute.unbind(dim=-1), inputh.unbind(dim=-1))):
				out, _state = net(inputu, inputhu, states[_tmp], _src_pad_mask, _chk_pad_mask, None, out, True)
				states[_tmp] = _state

			out = self.lsm(self.classifier(out)).view(bsize, beam_size, -1)

			_scores, _wds = out.topk(beam_size, dim=-1)
			_scores = (_scores.masked_fill(done_trans.unsqueeze(2).expand(bsize, beam_size, beam_size), 0.0) + sum_scores.unsqueeze(2).expand(bsize, beam_size, beam_size))

			if length_penalty > 0.0:
				lpv.masked_fill_(~done_trans.view(real_bsize, 1), ((step + 6.0) ** length_penalty) / lpv_base)

			if clip_beam and (length_penalty > 0.0):
				scores, _inds = (_scores.view(real_bsize, beam_size) / lpv.expand(real_bsize, beam_size)).view(bsize, beam_size2).topk(beam_size, dim=-1)
				_tinds = (_inds + torch.arange(0, bsizeb2, beam_size2, dtype=_inds.dtype, device=_inds.device).unsqueeze(1).expand_as(_inds)).view(real_bsize)
				sum_scores = _scores.view(bsizeb2).index_select(0, _tinds).view(bsize, beam_size)
			else:
				scores, _inds = _scores.view(bsize, beam_size2).topk(beam_size, dim=-1)
				_tinds = (_inds + torch.arange(0, bsizeb2, beam_size2, dtype=_inds.dtype, device=_inds.device).unsqueeze(1).expand_as(_inds)).view(real_bsize)
				sum_scores = scores

			wds = _wds.view(bsizeb2).index_select(0, _tinds).view(real_bsize, 1)

			_inds = (_inds // beam_size + torch.arange(0, real_bsize, beam_size, dtype=_inds.dtype, device=_inds.device).unsqueeze(1).expand_as(_inds)).view(real_bsize)

			trans = torch.cat((trans.index_select(0, _inds), wds.masked_fill(done_trans.view(real_bsize, 1), pad_id) if fill_pad else wds), 1)

			done_trans = (done_trans.view(real_bsize).index_select(0, _inds) | wds.eq(2).squeeze(1)).view(bsize, beam_size)

			_done = False
			if length_penalty > 0.0:
				lpv = lpv.index_select(0, _inds)
			elif (not return_all) and all_done(done_trans.select(1, 0), bsize):
				_done = True

			if _done or all_done(done_trans, real_bsize):
				break

			states = index_tensors(states, indices=_inds, dim=0)

		if (not clip_beam) and (length_penalty > 0.0):
			scores = scores / lpv.view(bsize, beam_size)
			scores, _inds = scores.topk(beam_size, dim=-1)
			_inds = (_inds + torch.arange(0, real_bsize, beam_size, dtype=_inds.dtype, device=_inds.device).unsqueeze(1).expand_as(_inds)).view(real_bsize)
			trans = trans.view(real_bsize, -1).index_select(0, _inds).view(bsize, beam_size, -1)

		if return_all:

			return trans, scores
		else:

			return trans.view(bsize, beam_size, -1).select(1, 0)

	def decode(self, inpute, inputh, src_pad_mask, chk_pad_mask, beam_size=1, max_len=512, length_penalty=0.0, fill_pad=False):

		return self.beam_decode(inpute, inputh, src_pad_mask, chk_pad_mask, beam_size, max_len, length_penalty, fill_pad=fill_pad) if beam_size > 1 else self.greedy_decode(inpute, inputh, src_pad_mask, chk_pad_mask, max_len, fill_pad=fill_pad)
