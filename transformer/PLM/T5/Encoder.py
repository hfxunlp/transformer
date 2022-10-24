#encoding: utf-8

import torch
from torch import nn
from modules.dropout import Dropout
from modules.norm import RMSNorm as Norm
from modules.plm.t5 import ResSelfAttn, PositionwiseFF
from utils.plm.base import copy_plm_parameter
from utils.plm.t5 import reorder_pemb
from cnfg.vocab.plm.t5 import pad_id
from cnfg.plm.t5.ihyp import *

from transformer.Encoder import EncoderLayer as EncoderLayerBase, Encoder as EncoderBase

class EncoderLayer(EncoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, ahsize=None, norm_residual=norm_residual_default, k_rel_pos=use_k_relative_position_encoder, max_bucket_distance=relative_position_max_bucket_distance_encoder, model_name="encoder", **kwargs):

		_ahsize = isize if ahsize is None else ahsize
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(EncoderLayer, self).__init__(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, ahsize=_ahsize, norm_residual=norm_residual, k_rel_pos=k_rel_pos, max_bucket_distance=max_bucket_distance, **kwargs)

		self.attn = ResSelfAttn(isize, _ahsize, num_head=num_head, dropout=attn_drop, norm_residual=norm_residual, k_rel_pos=k_rel_pos, max_bucket_distance=max_bucket_distance)
		self.ff = PositionwiseFF(isize, hsize=_fhsize, dropout=dropout, norm_residual=norm_residual, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, use_glu=use_glu_ffn)

	def load_plm(self, plm_parameters, model_name=None, layer_idx=None):

		_model_name = self.model_name if model_name is None else model_name
		with torch.no_grad():
			copy_plm_parameter(self.attn.net.adaptor.weight, plm_parameters, ["%s.block.%d.layer.0.SelfAttention.q.weight" % (_model_name, layer_idx,), "%s.block.%d.layer.0.SelfAttention.k.weight" % (_model_name, layer_idx,), "%s.block.%d.layer.0.SelfAttention.v.weight" % (_model_name, layer_idx,)], func=torch.cat, func_kwargs={"dim": 0})
			_bias_key = "%s.block.%d.layer.0.SelfAttention.q.bias" % (_model_name, layer_idx,)
			if self.attn.net.adaptor.bias is None and (_bias_key in plm_parameters):
				self.attn.net.adaptor.bias = nn.Parameter(torch.zeros(self.attn.net.adaptor.weight.size(0)))
			if self.attn.net.adaptor.bias is not None:
				copy_plm_parameter(self.attn.net.adaptor.bias, plm_parameters, [_bias_key, "%s.block.%d.layer.0.SelfAttention.k.bias" % (_model_name, layer_idx,), "%s.block.%d.layer.0.SelfAttention.v.bias" % (_model_name, layer_idx,)], func=torch.cat, func_kwargs={"dim": 0})
			copy_plm_parameter(self.attn.net.outer.weight, plm_parameters, "%s.block.%d.layer.0.SelfAttention.o.weight" % (_model_name, layer_idx,))
			_bias_key = "%s.block.%d.layer.0.SelfAttention.o.bias" % (_model_name, layer_idx,)
			if self.attn.net.outer.bias is None and (_bias_key in plm_parameters):
				self.attn.net.outer.bias = nn.Parameter(torch.zeros(self.attn.net.outer.weight.size(0)))
			if self.attn.net.outer.bias is not None:
				copy_plm_parameter(self.attn.net.outer.bias, plm_parameters, _bias_key)
			_bias_key = "%s.block.%d.layer.0.SelfAttention.relative_attention_bias.weight" % (_model_name, layer_idx,)
			if (self.attn.net.rel_pemb is not None) and (_bias_key in plm_parameters):
				copy_plm_parameter(self.attn.net.rel_pemb.weight, plm_parameters, _bias_key, func=reorder_pemb)
			copy_plm_parameter(self.attn.normer.weight, plm_parameters, "%s.block.%d.layer.0.layer_norm.weight" % (_model_name, layer_idx,))
			_bias_key = "%s.block.%d.layer.0.layer_norm.bias" % (_model_name, layer_idx,)
			if (self.attn.normer.bias is not None) and (_bias_key in plm_parameters):
				copy_plm_parameter(self.attn.normer.bias, plm_parameters, _bias_key)
			if use_glu_ffn is None:
				copy_plm_parameter(self.ff.net[0].weight, plm_parameters, "%s.block.%d.layer.1.DenseReluDense.wi.weight" % (_model_name, layer_idx,))
				_bias_key = "%s.block.%d.layer.1.DenseReluDense.wi.bias" % (_model_name, layer_idx,)
				if _bias_key in plm_parameters:
					copy_plm_parameter(self.ff.net[0].bias, plm_parameters, _bias_key)
			else:
				copy_plm_parameter(self.ff.net[0].weight, plm_parameters, ["%s.block.%d.layer.1.DenseReluDense.wi_0.weight" % (_model_name, layer_idx,), "%s.block.%d.layer.1.DenseReluDense.wi_1.weight" % (_model_name, layer_idx,)], func=torch.cat, func_kwargs={"dim": 0})
				_bias_key = "%s.block.%d.layer.1.DenseReluDense.wi_0.bias" % (_model_name, layer_idx,)
				if _bias_key in plm_parameters:
					copy_plm_parameter(self.ff.net[0].bias, plm_parameters, [_bias_key, "%s.block.%d.layer.1.DenseReluDense.wi_1.bias" % (_model_name, layer_idx,)], func=torch.cat, func_kwargs={"dim": 0})
			_l = self.ff.net[-2] if isinstance(self.ff.net[-1], Dropout) else self.ff.net[-1]
			copy_plm_parameter(_l.weight, plm_parameters, "%s.block.%d.layer.1.DenseReluDense.wo.weight" % (_model_name, layer_idx,))
			_bias_key = "%s.block.%d.layer.1.DenseReluDense.wo.bias" % (_model_name, layer_idx,)
			if _bias_key in plm_parameters:
				if _l.bias is None:
					_l.bias = nn.Parameter(torch.zeros(_l.weight.size(0)))
				copy_plm_parameter(_l.bias, plm_parameters, _bias_key)
			copy_plm_parameter(self.ff.normer.weight, plm_parameters, "%s.block.%d.layer.1.layer_norm.weight" % (_model_name, layer_idx,))
			_bias_key = "%s.block.%d.layer.1.layer_norm.bias" % (_model_name, layer_idx,)
			if (self.ff.normer.bias is not None) and (_bias_key in plm_parameters):
				copy_plm_parameter(self.ff.normer.bias, plm_parameters, _bias_key)

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, share_layer=False, disable_pemb=disable_std_pemb_encoder, model_name="encoder", **kwargs):

		_ahsize = isize if ahsize is None else ahsize
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindDecoderEmb=bindDecoderEmb, share_layer=share_layer, disable_pemb=disable_pemb, **kwargs)

		self.wemb.padding_idx = pad_id

		if share_layer:
			_shared_layer = EncoderLayer(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, ahsize=_ahsize, model_name=model_name)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([EncoderLayer(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, ahsize=_ahsize, k_rel_pos=use_k_relative_position_encoder if i == 0 else 0, max_bucket_distance=relative_position_max_bucket_distance_encoder if i == 0 else 0, model_name=model_name) for i in range(num_layer)])
		self.out_normer = Norm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters) if norm_output else None

	def forward(self, inputs, mask=None):

		out = self.wemb(inputs)

		if self.pemb is not None:
			out = out * sqrt(out.size(-1)) + self.pemb(inputs, expand=False)
		if self.drop is not None:
			out = self.drop(out)

		for net in self.nets:
			out = net(out, mask)

		if self.out_normer is not None:
			out = self.out_normer(out)
		if self.drop is not None:
			out = self.drop(out)

		return out

	def load_plm(self, plm_parameters, model_name=None, layer_idx=None):

		_model_name = self.model_name if model_name is None else model_name
		with torch.no_grad():
			_ = "%s.embed_tokens.weight" % _model_name
			copy_plm_parameter(self.wemb.weight, plm_parameters, _ if _ in plm_parameters else "shared.weight")
			copy_plm_parameter(self.out_normer.weight, plm_parameters, "%s.final_layer_norm.weight" % _model_name)
			_bias_key = "%s.final_layer_norm.bias" % _model_name
			if (self.out_normer.bias is not None) and (_bias_key in plm_parameters):
				copy_plm_parameter(self.out_normer.bias, plm_parameters, _bias_key)
			for i, net in enumerate(self.nets):
				net.load_plm(plm_parameters, model_name=_model_name, layer_idx=i)
