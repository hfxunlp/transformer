#encoding: utf-8

import torch
from torch import nn
from math import sqrt

class Decoder(nn.Module):

	# models: list of decoders

	def __init__(self, models):

		super(Decoder, self).__init__()

		self.nets = models

	# inpute: encoded representation from encoders [(bsize, seql, isize)...]
	# inputo: decoded translation (bsize, nquery)
	# src_pad_mask: mask for given encoding source sentence (bsize, 1, seql), see Encoder, generated with:
	#	src_pad_mask = input.eq(0).unsqueeze(1)

	def forward(self, inpute, inputo, src_pad_mask=None):

		bsize, nquery = inputo.size()

		outs = []

		_mask = self.nets[0]._get_subsequent_mask(inputo.size(1))

		# the following line of code is to mask <pad> for the decoder,
		# which I think is useless, since only <pad> may pay attention to previous <pad> tokens, whos loss will be omitted by the loss function.
		#_mask = torch.gt(_mask + inputo.eq(0).unsqueeze(1), 0)

		for model, inputu in zip(self.nets, inpute):

			out = model.wemb(inputo)

			out = out * sqrt(out.size(-1)) + model.pemb(inputo, expand=False)

			if model.drop is not None:
				out = model.drop(out)

			for net in model.nets:
				out = net(inputu, out, src_pad_mask, _mask)

			if model.out_normer is not None:
				out = model.out_normer(out)

			outs.append(model.lsm(model.classifier(out)))

		return torch.stack(outs).mean(0)

	# inpute: encoded representation from encoders [(bsize, seql, isize)...]
	# src_pad_mask: mask for given encoding source sentence (bsize, seql), see Encoder, get by:
	#	src_pad_mask = input.eq(0).unsqueeze(1)
	# beam_size: the beam size for beam search
	# max_len: maximum length to generate

	def decode(self, inpute, src_pad_mask, beam_size=1, max_len=512, length_penalty=0.0):

		return self.beam_decode(inpute, src_pad_mask, beam_size, max_len, length_penalty) if beam_size > 1 else self.greedy_decode(inpute, src_pad_mask, max_len)

	# inpute: encoded representation from encoders [(bsize, seql, isize)...]
	# src_pad_mask: mask for given encoding source sentence (bsize, 1, seql), see Encoder, generated with:
	#	src_pad_mask = input.eq(0).unsqueeze(1)
	# max_len: maximum length to generate

	def greedy_decode(self, inpute, src_pad_mask=None, max_len=512):

		bsize, seql, isize = inpute[0].size()

		sqrt_isize = sqrt(isize)

		outs = []

		for model, inputu in zip(self.nets, inpute):

			sos_emb = model.get_sos_emb(inputu)

			# out: input to the decoder for the first step (bsize, 1, isize)

			out = sos_emb * sqrt_isize + model.pemb.get_pos(0).view(1, 1, -1).expand(bsize, 1, -1)

			if model.drop is not None:
				out = model.drop(out)

			states = {}

			for _tmp, net in enumerate(model.nets):
				out, _state = net(inputu, None, src_pad_mask, None, out, True)
				states[_tmp] = _state

			if model.out_normer is not None:
				out = model.out_normer(out)

			# outs: [(bsize, 1, nwd)...]

			outs.append(model.lsm(model.classifier(out)))

		out = torch.stack(outs).mean(0)

		# wds: (bsize, 1)

		wds = torch.argmax(out, dim=-1)

		trans = [wds]

		# done_trans: (bsize)

		done_trans = wds.squeeze(1).eq(2)

		for i in range(1, max_len):

			outs = []

			for model, inputu in zip(self.nets, inpute):

				out = model.wemb(wds) * sqrt_isize + model.pemb.get_pos(i).view(1, 1, -1).expand(bsize, 1, -1)

				if model.drop is not None:
					out = model.drop(out)

				for _tmp, net in enumerate(model.nets):
					out, _state = net(inputu, states[_tmp], src_pad_mask, None, out, True)
					states[_tmp] = _state

				if model.out_normer is not None:
					out = model.out_normer(out)

				# outs: [(bsize, 1, nwd)...]
				outs.append(model.lsm(model.classifier(out)))

			out = torch.stack(outs).mean(0)

			wds = torch.argmax(out, dim=-1)

			trans.append(wds)

			done_trans = torch.gt(done_trans + wds.squeeze(1).eq(2), 0)
			if done_trans.sum().item() == bsize:
				break

		return torch.cat(trans, 1)

	# inpute: encoded representation from encoders [(bsize, seql, isize)...]
	# src_pad_mask: mask for given encoding source sentence (bsize, 1, seql), see Encoder, generated with:
	#	src_pad_mask = input.eq(0).unsqueeze(1)
	# beam_size: beam size
	# max_len: maximum length to generate

	def beam_decode(self, inpute, src_pad_mask=None, beam_size=8, max_len=512, length_penalty=0.0, return_all=False, clip_beam=False):

		bsize, seql, isize = inpute[0].size()

		beam_size2 = beam_size * beam_size
		bsizeb2 = bsize * beam_size2
		real_bsize = bsize * beam_size

		sqrt_isize = sqrt(isize)

		if length_penalty > 0.0:
			# lpv: length penalty vector for each beam (bsize * beam_size, 1)
			lpv = inpute[0].new_ones(real_bsize, 1)
			lpv_base = 6.0 ** length_penalty

		states = {}

		outs = []

		for _inum, (model, inputu) in enumerate(zip(self.nets, inpute)):

			out = model.get_sos_emb(inputu) * sqrt_isize + model.pemb.get_pos(0).view(1, 1, isize).expand(bsize, 1, isize)

			if model.drop is not None:
				out = model.drop(out)

			states[_inum] = {}

			for _tmp, net in enumerate(model.nets):
				out, _state = net(inputu, None, src_pad_mask, None, out, True)
				states[_inum][_tmp] = _state

			if model.out_normer is not None:
				out = model.out_normer(out)

			# outs: [(bsize, 1, nwd)]

			outs.append(model.lsm(model.classifier(out)))

		out = torch.stack(outs).mean(0)

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

		inpute = [inputu.repeat(1, beam_size, 1).view(real_bsize, seql, isize) for inputu in inpute]

		# _src_pad_mask: (bsize, 1, seql) => (bsize * beam_size, 1, seql)

		_src_pad_mask = None if src_pad_mask is None else src_pad_mask.repeat(1, beam_size, 1).view(real_bsize, 1, seql)

		# states[i][j]: (bsize, 1, isize) => (bsize * beam_size, 1, isize)

		for key, value in states.items():
			for _key, _value in value.items():
				value[_key] = _value.repeat(1, beam_size, 1).view(real_bsize, 1, isize)

		for step in range(1, max_len):

			outs = []

			for _inum, (model, inputu) in enumerate(zip(self.nets, inpute)):

				out = model.wemb(wds) * sqrt_isize + model.pemb.get_pos(step).view(1, 1, isize).expand(real_bsize, 1, isize)

				if model.drop is not None:
					out = model.drop(out)

				for _tmp, net in enumerate(model.nets):
					out, _state = net(inputu, states[_inum][_tmp], _src_pad_mask, None, out, True)
					states[_inum][_tmp] = _state

				if model.out_normer is not None:
					out = model.out_normer(out)

				# outs: [(bsize, beam_size, nwd)...]

				outs.append(model.lsm(model.classifier(out)).view(bsize, beam_size, -1))

			out = torch.stack(outs).mean(0)

			# find the top k ** 2 candidates and calculate route scores for them
			# _scores: (bsize, beam_size, beam_size)
			# done_trans: (bsize, beam_size)
			# scores: (bsize, beam_size)
			# _wds: (bsize, beam_size, beam_size)
			# mask_from_done_trans: (bsize, beam_size) => (bsize, beam_size * beam_size)
			# added_scores: (bsize, 1, beam_size) => (bsize, beam_size, beam_size)

			_scores, _wds = torch.topk(out, beam_size, dim=-1)
			_scores = (_scores.masked_fill(done_trans.unsqueeze(2).expand(bsize, beam_size, beam_size), 0.0) + scores.unsqueeze(2).expand(bsize, beam_size, beam_size))

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

			# update the corresponding hidden states
			# states[i][j]: (bsize * beam_size, nquery, isize)
			# _inds: (bsize, beam_size) => (bsize * beam_size)

			for key, value in states.items():
				for _key, _value in value.items():
					value[_key] = _value.index_select(0, _inds)

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

		# if length penalty is only applied in the last step, apply length penalty
		if (not clip_beam) and (length_penalty > 0.0):
			scores = scores / lpv.view(bsize, beam_size)

		if return_all:

			return trans, scores
		else:

			return trans.view(bsize, beam_size, -1).select(1, 0)
