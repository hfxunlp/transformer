#encoding: utf-8

import torch
from math import sqrt

from transformer.Decoder import Decoder as DecoderBase
from utils.base import index_tensors, select_zero_
from utils.decode.beam import expand_bsize_for_beam
from utils.sampler import SampleMax
from utils.torch.comp import all_done

from cnfg.ihyp import *
from cnfg.vocab.base import eos_id, pad_id

class Decoder(DecoderBase):

	# inpute: encoded representation from encoder (bsize, seql, isize, num_layer)
	def forward(self, inpute, inputo, src_pad_mask=None, **kwargs):

		bsize, nquery = inputo.size()

		out = self.wemb(inputo)

		if self.pemb is not None:
			out = self.pemb(inputo, expand=False).add(out, alpha=sqrt(out.size(-1)))
		if self.drop is not None:
			out = self.drop(out)

		_mask = self._get_subsequent_mask(nquery)

		for net, inputu in zip(self.nets, inpute.unbind(dim=-1)):
			out = net(inputu, out, src_pad_mask, _mask)

		if self.out_normer is not None:
			out = self.out_normer(out)

		out = self.lsm(self.classifier(out))

		return out

	# inpute: encoded representation from encoder (bsize, seql, isize, num_layer)
	def greedy_decode(self, inpute, src_pad_mask=None, max_len=512, fill_pad=False, sample=False, **kwargs):

		bsize = inpute.size(0)

		out = self.get_sos_emb(inpute)

		if self.pemb is not None:
			sqrt_isize = sqrt(out.size(-1))
			out = self.pemb.get_pos(0).add(out, alpha=sqrt_isize)
		if self.drop is not None:
			out = self.drop(out)

		states = {}

		for _tmp, (net, inputu) in enumerate(zip(self.nets, inpute.unbind(dim=-1))):
			out, _state = net(inputu, (None, None,), src_pad_mask, None, out)
			states[_tmp] = _state

		if self.out_normer is not None:
			out = self.out_normer(out)

		out = self.classifier(out)
		wds = SampleMax(out.softmax(-1), dim=-1, keepdim=False) if sample else out.argmax(dim=-1)

		trans = [wds]

		done_trans = wds.eq(eos_id)

		for i in range(1, max_len):

			out = self.wemb(wds)
			if self.pemb is not None:
				out = self.pemb.get_pos(i).add(out, alpha=sqrt_isize)
			if self.drop is not None:
				out = self.drop(out)

			for _tmp, (net, inputu) in enumerate(zip(self.nets, inpute.unbind(dim=-1))):
				out, _state = net(inputu, states[_tmp], src_pad_mask, None, out)
				states[_tmp] = _state

			if self.out_normer is not None:
				out = self.out_normer(out)

			out = self.classifier(out)
			wds = SampleMax(out.softmax(-1), dim=-1, keepdim=False) if sample else out.argmax(dim=-1)

			trans.append(wds.masked_fill(done_trans, pad_id) if fill_pad else wds)

			done_trans = done_trans | wds.eq(eos_id)
			if all_done(done_trans, bsize):
				break

		return torch.cat(trans, 1)

	# inpute: encoded representation from encoder (bsize, seql, isize, num_layer)
	def beam_decode(self, inpute, src_pad_mask=None, beam_size=8, max_len=512, length_penalty=0.0, return_all=False, clip_beam=clip_beam_with_lp, fill_pad=False, **kwargs):

		bsize, seql = inpute.size()[:2]

		beam_size2 = beam_size * beam_size
		bsizeb2 = bsize * beam_size2
		real_bsize = bsize * beam_size

		out = self.get_sos_emb(inpute)

		if length_penalty > 0.0:
			lpv = out.new_ones(real_bsize, 1)
			lpv_base = 6.0 ** length_penalty

		if self.pemb is not None:
			sqrt_isize = sqrt(out.size(-1))
			out = self.pemb.get_pos(0).add(out, alpha=sqrt_isize)
		if self.drop is not None:
			out = self.drop(out)

		states = {}

		for _tmp, (net, inputu) in enumerate(zip(self.nets, inpute.unbind(dim=-1))):
			out, _state = net(inputu, (None, None,), src_pad_mask, None, out)
			states[_tmp] = _state

		if self.out_normer is not None:
			out = self.out_normer(out)

		out = self.lsm(self.classifier(out))

		scores, wds = out.topk(beam_size, dim=-1)
		scores = scores.squeeze(1)
		sum_scores = scores
		wds = wds.view(real_bsize, 1)
		trans = wds
		_inds_add_beam2 = torch.arange(0, bsizeb2, beam_size2, dtype=wds.dtype, device=wds.device).unsqueeze(1).expand(bsize, beam_size)
		_inds_add_beam = torch.arange(0, real_bsize, beam_size, dtype=wds.dtype, device=wds.device).unsqueeze(1).expand(bsize, beam_size)

		done_trans = wds.view(bsize, beam_size).eq(eos_id)

		self.repeat_cross_attn_buffer(beam_size)

		_src_pad_mask = None if src_pad_mask is None else src_pad_mask.repeat(1, beam_size, 1).view(real_bsize, 1, seql)

		states = expand_bsize_for_beam(states, beam_size=beam_size)

		for step in range(1, max_len):

			out = self.wemb(wds)
			if self.pemb is not None:
				out = self.pemb.get_pos(step).add(out, alpha=sqrt_isize)
			if self.drop is not None:
				out = self.drop(out)

			for _tmp, (net, inputu) in enumerate(zip(self.nets, inpute.unbind(dim=-1))):
				out, _state = net(inputu, states[_tmp], _src_pad_mask, None, out)
				states[_tmp] = _state

			if self.out_normer is not None:
				out = self.out_normer(out)

			out = self.lsm(self.classifier(out)).view(bsize, beam_size, -1)

			_scores, _wds = out.topk(beam_size, dim=-1)
			_done_trans_unsqueeze = done_trans.unsqueeze(2)
			_scores = (_scores.masked_fill(_done_trans_unsqueeze.expand(bsize, beam_size, beam_size), 0.0) + sum_scores.unsqueeze(2).repeat(1, 1, beam_size).masked_fill_(select_zero_(_done_trans_unsqueeze.repeat(1, 1, beam_size), -1, 0), -inf_default))

			if length_penalty > 0.0:
				lpv.masked_fill_(~done_trans.view(real_bsize, 1), ((step + 6.0) ** length_penalty) / lpv_base)

			if clip_beam and (length_penalty > 0.0):
				scores, _inds = (_scores.view(real_bsize, beam_size) / lpv.expand(real_bsize, beam_size)).view(bsize, beam_size2).topk(beam_size, dim=-1)
				_tinds = (_inds + _inds_add_beam2).view(real_bsize)
				sum_scores = _scores.view(bsizeb2).index_select(0, _tinds).view(bsize, beam_size)
			else:
				scores, _inds = _scores.view(bsize, beam_size2).topk(beam_size, dim=-1)
				_tinds = (_inds + _inds_add_beam2).view(real_bsize)
				sum_scores = scores

			wds = _wds.view(bsizeb2).index_select(0, _tinds).view(real_bsize, 1)

			_inds = (_inds // beam_size + _inds_add_beam).view(real_bsize)

			trans = torch.cat((trans.index_select(0, _inds), wds.masked_fill(done_trans.view(real_bsize, 1), pad_id) if fill_pad else wds), 1)

			done_trans = (done_trans.view(real_bsize).index_select(0, _inds) | wds.eq(eos_id).squeeze(1)).view(bsize, beam_size)

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
			_inds = (_inds + _inds_add_beam).view(real_bsize)
			trans = trans.view(real_bsize, -1).index_select(0, _inds)

		if return_all:

			return trans.view(bsize, beam_size, -1), scores
		else:

			return trans.view(bsize, beam_size, -1).select(1, 0)
