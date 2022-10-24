#encoding: utf-8

import torch
from torch import nn
from modules.dropout import Dropout
from modules.plm.bert import PositionwiseFF
from math import sqrt

from utils.plm.base import copy_plm_parameter
from cnfg.vocab.plm.bert import pad_id
from cnfg.plm.bert.base import num_type
from cnfg.plm.bert.ihyp import *

from transformer.TA.Encoder import EncoderLayer as EncoderLayerBase
from transformer.Encoder import Encoder as EncoderBase

class EncoderLayer(EncoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, ahsize=None, norm_residual=norm_residual_default, k_rel_pos=use_k_relative_position_encoder, max_bucket_distance=relative_position_max_bucket_distance_encoder, model_name="bert", **kwargs):

		_ahsize = isize if ahsize is None else ahsize
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(EncoderLayer, self).__init__(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, ahsize=_ahsize, norm_residual=norm_residual, k_rel_pos=k_rel_pos, max_bucket_distance=max_bucket_distance, **kwargs)

		self.model_name = model_name
		self.ff = PositionwiseFF(isize, hsize=_fhsize, dropout=dropout, norm_residual=norm_residual, custom_act=use_adv_act_default, enable_bias=enable_prev_ln_bias_default, use_glu=use_glu_ffn)

	def load_plm(self, plm_parameters, model_name=None, layer_idx=None):

		_model_name = self.model_name if model_name is None else model_name
		with torch.no_grad():
			copy_plm_parameter(self.attn.net.adaptor.weight, plm_parameters, ["%s.encoder.layer.%d.attention.self.query.weight" % (_model_name, layer_idx,), "%s.encoder.layer.%d.attention.self.key.weight" % (_model_name, layer_idx,), "%s.encoder.layer.%d.attention.self.value.weight" % (_model_name, layer_idx,)], func=torch.cat, func_kwargs={"dim": 0})
			_bias_key = "%s.encoder.layer.%d.attention.self.query.bias" % (_model_name, layer_idx,)
			if self.attn.net.adaptor.bias is None and (_bias_key in plm_parameters):
				self.attn.net.adaptor.bias = nn.Parameter(torch.zeros(self.attn.net.adaptor.weight.size(0)))
			if self.attn.net.adaptor.bias is not None:
				copy_plm_parameter(self.attn.net.adaptor.bias, plm_parameters, [_bias_key, "%s.encoder.layer.%d.attention.self.key.bias" % (_model_name, layer_idx,), "%s.encoder.layer.%d.attention.self.value.bias" % (_model_name, layer_idx,)], func=torch.cat, func_kwargs={"dim": 0})
			copy_plm_parameter(self.attn.net.outer.weight, plm_parameters, "%s.encoder.layer.%d.attention.output.dense.weight" % (_model_name, layer_idx,))
			_bias_key = "%s.encoder.layer.%d.attention.output.dense.bias" % (_model_name, layer_idx,)
			if self.attn.net.outer.bias is None and (_bias_key in plm_parameters):
				self.attn.net.outer.bias = nn.Parameter(torch.zeros(self.attn.net.outer.weight.size(0)))
			if self.attn.net.outer.bias is not None:
				copy_plm_parameter(self.attn.net.outer.bias, plm_parameters, _bias_key)
			copy_plm_parameter(self.attn.normer.weight, plm_parameters, "%s.encoder.layer.%d.attention.output.LayerNorm.weight" % (_model_name, layer_idx,))
			copy_plm_parameter(self.attn.normer.bias, plm_parameters, "%s.encoder.layer.%d.attention.output.LayerNorm.bias" % (_model_name, layer_idx,))
			copy_plm_parameter(self.ff.net[0].weight, plm_parameters, "%s.encoder.layer.%d.intermediate.dense.weight" % (_model_name, layer_idx,))
			copy_plm_parameter(self.ff.net[0].bias, plm_parameters, "%s.encoder.layer.%d.intermediate.dense.bias" % (_model_name, layer_idx,))
			_l = self.ff.net[-2] if isinstance(self.ff.net[-1], Dropout) else self.ff.net[-1]
			copy_plm_parameter(_l.weight, plm_parameters, "%s.encoder.layer.%d.output.dense.weight" % (_model_name, layer_idx,))
			_bias_key = "%s.encoder.layer.%d.output.dense.bias" % (_model_name, layer_idx,)
			if _l.bias is None and (_bias_key in plm_parameters):
				_l.bias = nn.Parameter(torch.zeros(_l.weight.size(0)))
			if _l.bias is not None:
				copy_plm_parameter(_l.bias, plm_parameters, _bias_key)
			copy_plm_parameter(self.ff.normer.weight, plm_parameters, "%s.encoder.layer.%d.output.LayerNorm.weight" % (_model_name, layer_idx,))
			copy_plm_parameter(self.ff.normer.bias, plm_parameters, "%s.encoder.layer.%d.output.LayerNorm.bias" % (_model_name, layer_idx,))

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, num_type=num_type, share_layer=False, model_name="bert", **kwargs):

		_ahsize = isize if ahsize is None else ahsize
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, share_layer=share_layer, **kwargs)

		self.model_name = model_name
		self.pemb = nn.Parameter(torch.Tensor(xseql, isize).uniform_(- sqrt(2.0 / (isize + xseql)), sqrt(2.0 / (isize + xseql))))
		self.temb = nn.Embedding(num_type, isize)

		self.wemb.padding_idx = pad_id
		if share_layer:
			_shared_layer = EncoderLayer(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, ahsize=_ahsize, model_name=model_name)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([EncoderLayer(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, ahsize=_ahsize, model_name=model_name) for i in range(num_layer)])

	def forward(self, inputs, token_types=None, mask=None):

		seql = inputs.size(1)
		out = self.drop(self.out_normer(self.pemb.narrow(0, 0, seql) + (self.temb.weight[0] if token_types is None else self.temb(token_types)) + self.wemb(inputs)))

		_mask = inputs.eq(pad_id).unsqueeze(1) if mask is None else mask
		for net in self.nets:
			out = net(out, _mask)

		return out

	def load_plm(self, plm_parameters, model_name=None, layer_idx=None):

		_model_name = self.model_name if model_name is None else model_name
		with torch.no_grad():
			copy_plm_parameter(self.wemb.weight, plm_parameters, "%s.embeddings.word_embeddings.weight" % _model_name)
			copy_plm_parameter(self.pemb, plm_parameters, "%s.embeddings.position_embeddings.weight" % _model_name)
			copy_plm_parameter(self.temb.weight, plm_parameters, "%s.embeddings.token_type_embeddings.weight" % _model_name)
			copy_plm_parameter(self.out_normer.weight, plm_parameters, "%s.embeddings.LayerNorm.weight" % _model_name)
			copy_plm_parameter(self.out_normer.bias, plm_parameters, "%s.embeddings.LayerNorm.bias" % _model_name)
			for i, net in enumerate(self.nets):
				net.load_plm(plm_parameters, model_name=_model_name, layer_idx=i)

	def fix_init(self):

		super(Encoder, self).fix_init()
		with torch.no_grad():
			_ = sqrt(2.0 / sum(self.pemb.size()))
			self.pemb.uniform_(- _, _)
