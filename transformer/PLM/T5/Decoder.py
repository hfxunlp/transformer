#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from modules.dropout import Dropout
from modules.norm import RMSNorm as Norm
from modules.plm.t5 import PositionwiseFF, ResCrossAttn, ResSelfAttn
from transformer.Decoder import Decoder as DecoderBase, DecoderLayer as DecoderLayerBase
from utils.base import index_tensors, select_zero_
from utils.decode.beam import expand_bsize_for_beam
from utils.fmt.parser import parse_none
from utils.plm.base import copy_plm_parameter
from utils.plm.t5 import reorder_pemb
from utils.sampler import SampleMax
from utils.torch.comp import all_done, torch_no_grad

from cnfg.plm.t5.base import remove_classifier_bias
from cnfg.plm.t5.ihyp import *
from cnfg.vocab.plm.t5 import eos_id, pad_id, sos_id

class DecoderLayer(DecoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, norm_residual=norm_residual_default, k_rel_pos=use_k_relative_position_decoder, max_bucket_distance=relative_position_max_bucket_distance_decoder, k_rel_pos_cattn=use_k_relative_position_cattn, max_bucket_distance_cattn=relative_position_max_bucket_distance_cattn, model_name="decoder", **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(DecoderLayer, self).__init__(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, norm_residual=norm_residual, k_rel_pos=k_rel_pos, max_bucket_distance=max_bucket_distance, **kwargs)

		self.model_name = model_name
		self.self_attn = ResSelfAttn(isize, _ahsize, num_head=num_head, dropout=attn_drop, norm_residual=norm_residual, k_rel_pos=k_rel_pos, uni_direction_reduction=True, max_bucket_distance=max_bucket_distance)
		self.cross_attn = ResCrossAttn(isize, _ahsize, num_head=num_head, dropout=attn_drop, norm_residual=norm_residual, k_rel_pos=k_rel_pos_cattn, max_bucket_distance=max_bucket_distance_cattn)
		self.ff = PositionwiseFF(isize, hsize=_fhsize, dropout=dropout, act_drop=act_drop, norm_residual=norm_residual, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, use_glu=use_glu_ffn)

	def forward(self, inpute, inputo, src_pad_mask=None, tgt_pad_mask=None, query_unit=None, **kwargs):

		if query_unit is None:
			context = self.self_attn(inputo, mask=tgt_pad_mask)
		else:
			context, states_return = self.self_attn(query_unit, states=inputo)

		context = self.cross_attn(context, inpute, mask=src_pad_mask, step=0 if query_unit is None else states_return[0].size(-1))

		context = self.ff(context)

		if query_unit is None:
			return context
		else:
			return context, states_return

	def load_plm(self, plm_parameters, model_name=None, layer_idx=None, **kwargs):

		_model_name = parse_none(model_name, self.model_name)
		with torch_no_grad():
			copy_plm_parameter(self.self_attn.net.adaptor.weight, plm_parameters, ["%s.block.%d.layer.0.SelfAttention.q.weight" % (_model_name, layer_idx,), "%s.block.%d.layer.0.SelfAttention.k.weight" % (_model_name, layer_idx,), "%s.block.%d.layer.0.SelfAttention.v.weight" % (_model_name, layer_idx,)], func=torch.cat, func_kwargs={"dim": 0})
			_bias_key = "%s.block.%d.layer.0.SelfAttention.q.bias" % (_model_name, layer_idx,)
			if (self.self_attn.net.adaptor.bias is None) and (_bias_key in plm_parameters):
				self.self_attn.net.adaptor.bias = nn.Parameter(torch.zeros(self.attn.net.adaptor.weight.size(0)))
			if self.self_attn.net.adaptor.bias is not None:
				copy_plm_parameter(self.self_attn.net.adaptor.bias, plm_parameters, [_bias_key, "%s.block.%d.layer.0.SelfAttention.k.bias" % (_model_name, layer_idx,), "%s.block.%d.layer.0.SelfAttention.v.bias" % (_model_name, layer_idx,)], func=torch.cat, func_kwargs={"dim": 0})
			copy_plm_parameter(self.self_attn.net.outer.weight, plm_parameters, "%s.block.%d.layer.0.SelfAttention.o.weight" % (_model_name, layer_idx,))
			_bias_key = "%s.block.%d.layer.0.SelfAttention.o.bias" % (_model_name, layer_idx,)
			if (self.self_attn.net.outer.bias is None) and (_bias_key in plm_parameters):
				self.self_attn.net.outer.bias = nn.Parameter(torch.zeros(self.attn.net.outer.weight.size(0)))
			if self.self_attn.net.outer.bias is not None:
				copy_plm_parameter(self.self_attn.net.outer.bias, plm_parameters, _bias_key)
			_bias_key = "%s.block.%d.layer.0.SelfAttention.relative_attention_bias.weight" % (_model_name, layer_idx,)
			if (self.self_attn.net.rel_pemb is not None) and (_bias_key in plm_parameters):
				copy_plm_parameter(self.self_attn.net.rel_pemb.weight, plm_parameters, _bias_key)
			copy_plm_parameter(self.self_attn.normer.weight, plm_parameters, "%s.block.%d.layer.0.layer_norm.weight" % (_model_name, layer_idx,))
			_bias_key = "%s.block.%d.layer.0.layer_norm.bias" % (_model_name, layer_idx,)
			if (self.self_attn.normer.bias is not None) and (_bias_key in plm_parameters):
				copy_plm_parameter(self.self_attn.normer.bias, plm_parameters, _bias_key)
			copy_plm_parameter(self.cross_attn.net.query_adaptor.weight, plm_parameters, "%s.block.%d.layer.1.EncDecAttention.q.weight" % (_model_name, layer_idx,))
			_bias_key = "%s.block.%d.layer.1.EncDecAttention.q.bias" % (_model_name, layer_idx,)
			if (self.cross_attn.net.query_adaptor.bias is None) and (_bias_key in plm_parameters):
				self.cross_attn.net.query_adaptor.bias = nn.Parameter(torch.zeros(self.cross_attn.net.query_adaptor.weight.size(0)))
			if self.cross_attn.net.query_adaptor.bias is not None:
				copy_plm_parameter(self.cross_attn.net.query_adaptor.bias, plm_parameters, _bias_key)
			copy_plm_parameter(self.cross_attn.net.kv_adaptor.weight, plm_parameters, ["%s.block.%d.layer.1.EncDecAttention.k.weight" % (_model_name, layer_idx,), "%s.block.%d.layer.1.EncDecAttention.v.weight" % (_model_name, layer_idx,)], func=torch.cat, func_kwargs={"dim": 0})
			_bias_key = "%s.block.%d.layer.1.EncDecAttention.k.bias" % (_model_name, layer_idx,)
			if (self.cross_attn.net.kv_adaptor.bias is None) and (_bias_key in plm_parameters):
				self.cross_attn.net.kv_adaptor.bias = nn.Parameter(torch.zeros(self.cross_attn.net.kv_adaptor.weight.size(0)))
			if self.cross_attn.net.kv_adaptor.bias is not None:
				copy_plm_parameter(self.cross_attn.net.kv_adaptor.bias, plm_parameters, [_bias_key, "%s.block.%d.layer.1.EncDecAttention.v.bias" % (_model_name, layer_idx,)], func=torch.cat, func_kwargs={"dim": 0})
			copy_plm_parameter(self.cross_attn.net.outer.weight, plm_parameters, "%s.block.%d.layer.1.EncDecAttention.o.weight" % (_model_name, layer_idx,))
			_bias_key = "%s.block.%d.layer.1.EncDecAttention.o.bias" % (_model_name, layer_idx,)
			if (self.cross_attn.net.outer.bias is None) and (_bias_key in plm_parameters):
				self.cross_attn.net.outer.bias = nn.Parameter(torch.zeros(self.cross_attn.net.outer.weight.size(0)))
			if self.cross_attn.net.outer.bias is not None:
				copy_plm_parameter(self.cross_attn.net.outer.bias, plm_parameters, _bias_key)
			_bias_key = "%s.block.%d.layer.1.EncDecAttention.relative_attention_bias.weight" % (_model_name, layer_idx,)
			if (self.cross_attn.net.rel_pemb is not None) and (_bias_key in plm_parameters):
				copy_plm_parameter(self.cross_attn.net.rel_pemb.weight, plm_parameters, _bias_key, func=reorder_pemb)
			copy_plm_parameter(self.cross_attn.normer.weight, plm_parameters, "%s.block.%d.layer.1.layer_norm.weight" % (_model_name, layer_idx,))
			_bias_key = "%s.block.%d.layer.1.layer_norm.bias" % (_model_name, layer_idx,)
			if (self.cross_attn.normer.bias is not None) and (_bias_key in plm_parameters):
				copy_plm_parameter(self.cross_attn.normer.bias, plm_parameters, _bias_key)
			if use_glu_ffn is None:
				copy_plm_parameter(self.ff.net[0].weight, plm_parameters, "%s.block.%d.layer.2.DenseReluDense.wi.weight" % (_model_name, layer_idx,))
				_bias_key = "%s.block.%d.layer.2.DenseReluDense.wi.bias" % (_model_name, layer_idx,)
				if _bias_key in plm_parameters:
					copy_plm_parameter(self.ff.net[0].bias, plm_parameters, _bias_key)
			else:
				copy_plm_parameter(self.ff.net[0].weight, plm_parameters, ["%s.block.%d.layer.2.DenseReluDense.wi_0.weight" % (_model_name, layer_idx,), "%s.block.%d.layer.2.DenseReluDense.wi_1.weight" % (_model_name, layer_idx,)], func=torch.cat, func_kwargs={"dim": 0})
				_bias_key = "%s.block.%d.layer.2.DenseReluDense.wi_0.bias" % (_model_name, layer_idx,)
				if _bias_key in plm_parameters:
					copy_plm_parameter(self.ff.net[0].bias, plm_parameters, [_bias_key, "%s.block.%d.layer.2.DenseReluDense.wi_1.bias" % (_model_name, layer_idx,)], func=torch.cat, func_kwargs={"dim": 0})
			_l = self.ff.net[-2] if isinstance(self.ff.net[-1], Dropout) else self.ff.net[-1]
			copy_plm_parameter(_l.weight, plm_parameters, "%s.block.%d.layer.2.DenseReluDense.wo.weight" % (_model_name, layer_idx,))
			_bias_key = "%s.block.%d.layer.2.DenseReluDense.wo.bias" % (_model_name, layer_idx,)
			if _bias_key in plm_parameters:
				if _l.bias is None:
					_l.bias = nn.Parameter(torch.zeros(_l.weight.size(0)))
				copy_plm_parameter(_l.bias, plm_parameters, _bias_key)
			copy_plm_parameter(self.ff.normer.weight, plm_parameters, "%s.block.%d.layer.2.layer_norm.weight" % (_model_name, layer_idx,))
			_bias_key = "%s.block.%d.layer.2.layer_norm.bias" % (_model_name, layer_idx,)
			if (self.ff.normer.bias is not None) and (_bias_key in plm_parameters):
				copy_plm_parameter(self.ff.normer.bias, plm_parameters, _bias_key)

class Decoder(DecoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, emb_w=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindemb=True, forbidden_index=None, share_layer=False, disable_pemb=disable_std_pemb_decoder, model_name="decoder", **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Decoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindemb=bindemb, forbidden_index=forbidden_index, share_layer=share_layer, disable_pemb=disable_pemb, **kwargs)

		self.model_name = model_name
		self.wemb.padding_idx = pad_id

		if share_layer:
			_shared_layer = DecoderLayer(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, model_name=model_name)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([DecoderLayer(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, k_rel_pos=use_k_relative_position_decoder, max_bucket_distance=relative_position_max_bucket_distance_decoder, k_rel_pos_cattn=use_k_relative_position_cattn, max_bucket_distance_cattn=relative_position_max_bucket_distance_cattn, model_name=model_name) for i in range(num_layer)])# if i == 0 else 0
		self.out_normer = Norm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters) if norm_output else None

	def forward(self, inpute, inputo, src_pad_mask=None, word_prediction=False, **kwargs):

		nquery = inputo.size(-1)

		out = self.wemb(inputo)

		if self.pemb is not None:
			out = self.pemb(inputo, expand=False).add(out, alpha=sqrt(out.size(-1)))
		if self.drop is not None:
			out = self.drop(out)

		_mask = self._get_subsequent_mask(nquery)

		for net in self.nets:
			out = net(inpute, out, src_pad_mask, _mask)

		if self.out_normer is not None:
			out = self.out_normer(out)
		if self.drop is not None:
			out = self.drop(out)
		if word_prediction:
			# Rescaling output before token prediction
			# https://github.com/huggingface/transformers/blob/v4.22.2/src/transformers/models/t5/modeling_t5.py#L1674
			# https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
			out = self.lsm(self.classifier(out * sqrt(out.size(-1))))

		return out

	def greedy_decode(self, inpute, src_pad_mask=None, max_len=512, fill_pad=False, sample=False, **kwargs):

		bsize = inpute.size(0)

		out = self.get_sos_emb(inpute)

		sqrt_isize = sqrt(out.size(-1))
		if self.pemb is not None:
			out = self.pemb.get_pos(0).add(out, alpha=sqrt_isize)
		if self.drop is not None:
			out = self.drop(out)

		states = {}

		for _tmp, net in enumerate(self.nets):
			out, _state = net(inpute, (None, None,), src_pad_mask, None, out)
			states[_tmp] = _state

		if self.out_normer is not None:
			out = self.out_normer(out)
		if self.drop is not None:
			out = self.drop(out)

		out = self.classifier(out * sqrt_isize)
		wds = SampleMax(out.softmax(-1), dim=-1, keepdim=False) if sample else out.argmax(dim=-1)
		trans = [wds]
		done_trans = wds.eq(eos_id)

		for i in range(1, max_len):

			out = self.wemb(wds)
			if self.pemb is not None:
				out = self.pemb.get_pos(i).add(out, alpha=sqrt_isize)
			if self.drop is not None:
				out = self.drop(out)

			for _tmp, net in enumerate(self.nets):
				out, _state = net(inpute, states[_tmp], src_pad_mask, None, out)
				states[_tmp] = _state

			if self.out_normer is not None:
				out = self.out_normer(out)
			if self.drop is not None:
				out = self.drop(out)

			out = self.classifier(out * sqrt_isize)
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

		sqrt_isize = sqrt(out.size(-1))
		if self.pemb is not None:
			out = self.pemb.get_pos(0).add(out, alpha=sqrt_isize)
		if self.drop is not None:
			out = self.drop(out)

		states = {}

		for _tmp, net in enumerate(self.nets):
			out, _state = net(inpute, (None, None,), src_pad_mask, None, out)
			states[_tmp] = _state

		if self.out_normer is not None:
			out = self.out_normer(out)
		if self.drop is not None:
			out = self.drop(out)

		out = self.lsm(self.classifier(out * sqrt_isize))

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

			for _tmp, net in enumerate(self.nets):
				out, _state = net(inpute, states[_tmp], _src_pad_mask, None, out)
				states[_tmp] = _state

			if self.out_normer is not None:
				out = self.out_normer(out)
			if self.drop is not None:
				out = self.drop(out)

			out = self.lsm(self.classifier(out * sqrt_isize)).view(bsize, beam_size, -1)

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

	def get_sos_emb(self, inpute, bsize=None):

		bsize = inpute.size(0) if bsize is None else bsize

		return self.wemb.weight[sos_id].view(1, 1, -1).expand(bsize, 1, -1)

	def fix_init(self):

		self.fix_load()
		with torch_no_grad():
			#self.wemb.weight[pad_id].zero_()
			self.classifier.weight[pad_id].zero_()

	def load_plm(self, plm_parameters, model_name=None, **kwargs):

		_model_name = parse_none(model_name, self.model_name)
		with torch_no_grad():
			if "lm_head.weight" in plm_parameters:
				copy_plm_parameter(self.classifier.weight, plm_parameters, "lm_head.weight")
			_ = "%s.embed_tokens.weight" % _model_name
			copy_plm_parameter(self.wemb.weight, plm_parameters, _ if _ in plm_parameters else "shared.weight")
			copy_plm_parameter(self.out_normer.weight, plm_parameters, "%s.final_layer_norm.weight" % _model_name)
			_bias_key = "%s.final_layer_norm.bias" % _model_name
			if (self.out_normer.bias is not None) and (_bias_key in plm_parameters):
				copy_plm_parameter(self.out_normer.bias, plm_parameters, _bias_key)
			for i, net in enumerate(self.nets):
				net.load_plm(plm_parameters, model_name=_model_name, layer_idx=i, **kwargs)
		# T5 does NOT have the bias vector in the classifier
		if remove_classifier_bias:
			self.classifier.bias = None
