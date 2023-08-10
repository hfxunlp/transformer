#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from modules.TA import PositionwiseFF, ResCrossAttn, ResSelfAttn
from transformer.Decoder import Decoder as DecoderBase, DecoderLayer as DecoderLayerBase
from utils.base import index_tensors, select_zero_
from utils.decode.beam import expand_bsize_for_beam
from utils.fmt.parser import parse_none
from utils.plm.bart import load_plm_decoder_layer
from utils.plm.base import copy_plm_parameter
from utils.sampler import SampleMax
from utils.torch.comp import all_done, torch_no_grad

from cnfg.plm.bart.base import remove_classifier_bias
from cnfg.plm.bart.ihyp import *
from cnfg.vocab.plm.roberta import eos_id, pad_id, pemb_start_ind

class DecoderLayer(DecoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, norm_residual=norm_residual_default, k_rel_pos=use_k_relative_position_decoder, max_bucket_distance=relative_position_max_bucket_distance_decoder, model_name="decoder", **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(DecoderLayer, self).__init__(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, norm_residual=norm_residual, k_rel_pos=k_rel_pos, max_bucket_distance=max_bucket_distance, **kwargs)

		self.model_name = model_name
		self.self_attn = ResSelfAttn(isize, _ahsize, num_head=num_head, dropout=attn_drop, norm_residual=norm_residual, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default, k_rel_pos=k_rel_pos, uni_direction_reduction=True, max_bucket_distance=max_bucket_distance, xseql=cache_len_default)
		self.cross_attn = ResCrossAttn(isize, _ahsize, num_head=num_head, dropout=attn_drop, norm_residual=norm_residual, enable_bias=enable_prev_ln_bias_default, enable_proj_bias=enable_proj_bias_default)
		self.ff = PositionwiseFF(isize, hsize=_fhsize, dropout=dropout, act_drop=act_drop, norm_residual=norm_residual, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, use_glu=use_glu_ffn)

	def load_plm(self, plm_parameters, model_name=None, layer_idx=None, **kwargs):

		load_plm_decoder_layer(self, plm_parameters, model_name=model_name, layer_idx=layer_idx, **kwargs)

class Decoder(DecoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, emb_w=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindemb=True, forbidden_index=None, share_layer=False, disable_pemb=disable_std_pemb_decoder, model_name="decoder", **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Decoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindemb=bindemb, forbidden_index=forbidden_index, share_layer=share_layer, disable_pemb=disable_pemb, **kwargs)

		self.model_name = model_name
		self.wemb.padding_idx = pad_id
		self.pemb = None if disable_pemb else nn.Parameter(torch.Tensor(xseql, isize).uniform_(- sqrt(2.0 / (isize + xseql)), sqrt(2.0 / (isize + xseql))))

		if share_layer:
			_shared_layer = DecoderLayer(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, model_name=model_name)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([DecoderLayer(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, model_name=model_name) for i in range(num_layer)])

	def forward(self, inpute, inputo, src_pad_mask=None, word_prediction=False, **kwargs):

		nquery = inputo.size(-1)

		out = self.wemb(inputo)
		if self.pemb is not None:
			out = out + self.pemb.narrow(0, pemb_start_ind, nquery)
		if self.out_normer is not None:
			out = self.out_normer(out)
		if self.drop is not None:
			out = self.drop(out)

		_mask = self._get_subsequent_mask(nquery)

		for net in self.nets:
			out = net(inpute, out, src_pad_mask, _mask)

		if word_prediction:
			out = self.lsm(self.classifier(out))

		return out

	def greedy_decode(self, inpute, src_pad_mask=None, max_len=512, fill_pad=False, sample=False, **kwargs):

		bsize = inpute.size(0)

		out = self.get_sos_emb(inpute)
		if self.pemb is not None:
			out = out + self.pemb[pemb_start_ind]
		if self.out_normer is not None:
			out = self.out_normer(out)
		if self.drop is not None:
			out = self.drop(out)

		states = {}

		for _tmp, net in enumerate(self.nets):
			out, _state = net(inpute, (None, None,), src_pad_mask, None, out)
			states[_tmp] = _state

		out = self.classifier(out)
		wds = SampleMax(out.softmax(-1), dim=-1, keepdim=False) if sample else out.argmax(dim=-1)

		trans = [wds]
		done_trans = wds.eq(eos_id)

		for i in range(1, max_len):

			out = self.wemb(wds)
			if self.pemb is not None:
				out = out + self.pemb[pemb_start_ind + i]
			if self.out_normer is not None:
				out = self.out_normer(out)
			if self.drop is not None:
				out = self.drop(out)

			for _tmp, net in enumerate(self.nets):
				out, _state = net(inpute, states[_tmp], src_pad_mask, None, out)
				states[_tmp] = _state

			out = self.classifier(out)
			wds = SampleMax(out.softmax(-1), dim=-1, keepdim=False) if sample else out.argmax(dim=-1)

			trans.append(wds.masked_fill(done_trans, pad_id) if fill_pad else wds)

			done_trans = done_trans | wds.eq(eos_id)
			if all_done(done_trans, bsize):
				break

		return torch.cat(trans, 1)

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
			out = out + self.pemb[pemb_start_ind]
		if self.out_normer is not None:
			out = self.out_normer(out)
		if self.drop is not None:
			out = self.drop(out)

		states = {}

		for _tmp, net in enumerate(self.nets):
			out, _state = net(inpute, (None, None,), src_pad_mask, None, out)
			states[_tmp] = _state

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
				out = out + self.pemb[pemb_start_ind + step]
			if self.out_normer is not None:
				out = self.out_normer(out)
			if self.drop is not None:
				out = self.drop(out)

			for _tmp, net in enumerate(self.nets):
				out, _state = net(inpute, states[_tmp], _src_pad_mask, None, out)
				states[_tmp] = _state

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

	# BART starts decoding with the <eos> token
	def get_sos_emb(self, inpute, bsize=None):

		bsize = inpute.size(0) if bsize is None else bsize

		return self.wemb.weight[eos_id].view(1, 1, -1).expand(bsize, 1, -1)

	def fix_init(self):

		self.fix_load()
		with torch_no_grad():
			#self.wemb.weight[pad_id].zero_()
			self.classifier.weight[pad_id].zero_()
			if self.pemb is not None:
				_ = sqrt(2.0 / sum(self.pemb.size()))
				self.pemb.uniform_(- _, _)

	def load_plm(self, plm_parameters, model_name=None, **kwargs):

		_model_name = parse_none(model_name, self.model_name)
		with torch_no_grad():
			copy_plm_parameter(self.wemb.weight, plm_parameters, "%s.embed_tokens.weight" % _model_name)
			copy_plm_parameter(self.pemb, plm_parameters, "%s.embed_positions.weight" % _model_name)
			copy_plm_parameter(self.out_normer.weight, plm_parameters, "%s.layernorm_embedding.weight" % _model_name)
			copy_plm_parameter(self.out_normer.bias, plm_parameters, "%s.layernorm_embedding.bias" % _model_name)
			if (not remove_classifier_bias) and ("final_logits_bias" in plm_parameters):
				if self.classifier.bias is None:
					self.classifier.bias = nn.Parameter(torch.zeros(self.classifier.weight.size(0)))
				copy_plm_parameter(self.classifier.bias, plm_parameters, "final_logits_bias")
			for i, net in enumerate(self.nets):
				net.load_plm(plm_parameters, model_name=_model_name, layer_idx=i, **kwargs)
		# BART does NOT have the bias vector in the classifier
		if remove_classifier_bias:
			self.classifier.bias = None
