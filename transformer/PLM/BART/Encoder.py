#encoding: utf-8

import torch
from torch import nn
from modules.dropout import Dropout
from modules.TA import PositionwiseFF
from utils.plm.base import copy_plm_parameter
from cnfg.vocab.plm.roberta import pad_id, pemb_start_ind
from cnfg.plm.bart.ihyp import *

from transformer.PLM.BERT.Encoder import EncoderLayer as EncoderLayerBase, Encoder as EncoderBase

class EncoderLayer(EncoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, ahsize=None, norm_residual=norm_residual_default, k_rel_pos=use_k_relative_position_encoder, max_bucket_distance=relative_position_max_bucket_distance_encoder, model_name="encoder", **kwargs):

		_ahsize = isize if ahsize is None else ahsize
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(EncoderLayer, self).__init__(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, ahsize=_ahsize, norm_residual=norm_residual, k_rel_pos=k_rel_pos, max_bucket_distance=max_bucket_distance, model_name=model_name, **kwargs)

		self.ff = PositionwiseFF(isize, hsize=_fhsize, dropout=dropout, norm_residual=norm_residual, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, use_glu=use_glu_ffn)

	def load_plm(self, plm_parameters, model_name=None, layer_idx=None):

		_model_name = self.model_name if model_name is None else model_name
		with torch.no_grad():
			copy_plm_parameter(self.attn.net.adaptor.weight, plm_parameters, ["%s.layers.%d.self_attn.q_proj.weight" % (_model_name, layer_idx,), "%s.layers.%d.self_attn.k_proj.weight" % (_model_name, layer_idx,), "%s.layers.%d.self_attn.v_proj.weight" % (_model_name, layer_idx,)], func=torch.cat, func_kwargs={"dim": 0})
			_bias_key = "%s.layers.%d.self_attn.q_proj.bias" % (_model_name, layer_idx,)
			if self.attn.net.adaptor.bias is None and (_bias_key in plm_parameters):
				self.attn.net.adaptor.bias = nn.Parameter(torch.zeros(self.attn.net.adaptor.weight.size(0)))
			if self.attn.net.adaptor.bias is not None:
				copy_plm_parameter(self.attn.net.adaptor.bias, plm_parameters, [_bias_key, "%s.layers.%d.self_attn.k_proj.bias" % (_model_name, layer_idx,), "%s.layers.%d.self_attn.v_proj.bias" % (_model_name, layer_idx,)], func=torch.cat, func_kwargs={"dim": 0})
			copy_plm_parameter(self.attn.net.outer.weight, plm_parameters, "%s.layers.%d.self_attn.out_proj.weight" % (_model_name, layer_idx,))
			_bias_key = "%s.layers.%d.self_attn.out_proj.bias" % (_model_name, layer_idx,)
			if self.attn.net.outer.bias is None and (_bias_key in plm_parameters):
				self.attn.net.outer.bias = nn.Parameter(torch.zeros(self.attn.net.outer.weight.size(0)))
			if self.attn.net.outer.bias is not None:
				copy_plm_parameter(self.attn.net.outer.bias, plm_parameters, _bias_key)
			copy_plm_parameter(self.attn.normer.weight, plm_parameters, "%s.layers.%d.self_attn_layer_norm.weight" % (_model_name, layer_idx,))
			copy_plm_parameter(self.attn.normer.bias, plm_parameters, "%s.layers.%d.self_attn_layer_norm.bias" % (_model_name, layer_idx,))
			copy_plm_parameter(self.ff.net[0].weight, plm_parameters, "%s.layers.%d.fc1.weight" % (_model_name, layer_idx,))
			copy_plm_parameter(self.ff.net[0].bias, plm_parameters, "%s.layers.%d.fc1.bias" % (_model_name, layer_idx,))
			_l = self.ff.net[-2] if isinstance(self.ff.net[-1], Dropout) else self.ff.net[-1]
			copy_plm_parameter(_l.weight, plm_parameters, "%s.layers.%d.fc2.weight" % (_model_name, layer_idx,))
			_bias_key = "%s.layers.%d.fc2.bias" % (_model_name, layer_idx,)
			if _l.bias is None and (_bias_key in plm_parameters):
				_l.bias = nn.Parameter(torch.zeros(_l.weight.size(0)))
			if _l.bias is not None:
				copy_plm_parameter(_l.bias, plm_parameters, _bias_key)
			copy_plm_parameter(self.ff.normer.weight, plm_parameters, "%s.layers.%d.final_layer_norm.weight" % (_model_name, layer_idx,))
			copy_plm_parameter(self.ff.normer.bias, plm_parameters, "%s.layers.%d.final_layer_norm.bias" % (_model_name, layer_idx,))

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, share_layer=False, model_name="encoder", **kwargs):

		_ahsize = isize if ahsize is None else ahsize
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindDecoderEmb=bindDecoderEmb, share_layer=share_layer, model_name=model_name, **kwargs)

		self.wemb.padding_idx = pad_id
		self.temb = None

		if share_layer:
			_shared_layer = EncoderLayer(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, ahsize=_ahsize, model_name=model_name)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([EncoderLayer(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, ahsize=_ahsize, model_name=model_name) for i in range(num_layer)])

	def forward(self, inputs, mask=None, **kwargs):

		seql = inputs.size(1)
		out = self.drop(self.out_normer(self.pemb.narrow(0, pemb_start_ind, seql) + self.wemb(inputs)))

		_mask = inputs.eq(pad_id).unsqueeze(1) if mask is None else mask
		for net in self.nets:
			out = net(out, _mask)

		return out

	def load_plm(self, plm_parameters, model_name=None, layer_idx=None):

		_model_name = self.model_name if model_name is None else model_name
		with torch.no_grad():
			copy_plm_parameter(self.wemb.weight, plm_parameters, "%s.embed_tokens.weight" % _model_name)
			copy_plm_parameter(self.pemb, plm_parameters, "%s.embed_positions.weight" % _model_name)
			copy_plm_parameter(self.out_normer.weight, plm_parameters, "%s.layernorm_embedding.weight" % _model_name)
			copy_plm_parameter(self.out_normer.bias, plm_parameters, "%s.layernorm_embedding.bias" % _model_name)
			for i, net in enumerate(self.nets):
				net.load_plm(plm_parameters, model_name=_model_name, layer_idx=i)
