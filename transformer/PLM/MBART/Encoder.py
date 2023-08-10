#encoding: utf-8

import torch
from math import sqrt
from torch import nn

from modules.plm.mbart import PositionwiseFF, ResSelfAttn
from transformer.Encoder import Encoder as EncoderBase, EncoderLayer as EncoderLayerBase
from utils.fmt.parser import parse_none
from utils.plm.bart import load_plm_encoder_layer
from utils.plm.base import copy_plm_parameter
from utils.torch.comp import torch_no_grad

from cnfg.plm.mbart.ihyp import *
from cnfg.vocab.plm.mbart import pad_id, pemb_start_ind

class EncoderLayer(EncoderLayerBase):

	def __init__(self, isize, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, ahsize=None, norm_residual=norm_residual_default, k_rel_pos=use_k_relative_position_encoder, max_bucket_distance=relative_position_max_bucket_distance_encoder, model_name="model.encoder", **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(EncoderLayer, self).__init__(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, norm_residual=norm_residual, k_rel_pos=k_rel_pos, max_bucket_distance=max_bucket_distance, **kwargs)

		self.model_name = model_name
		self.attn = ResSelfAttn(isize, _ahsize, num_head=num_head, dropout=attn_drop, norm_residual=norm_residual, k_rel_pos=k_rel_pos, max_bucket_distance=max_bucket_distance)
		self.ff = PositionwiseFF(isize, hsize=_fhsize, dropout=dropout, act_drop=act_drop, norm_residual=norm_residual)

	def load_plm(self, plm_parameters, model_name=None, layer_idx=None, **kwargs):

		load_plm_encoder_layer(self, plm_parameters, model_name=model_name, layer_idx=layer_idx, **kwargs)

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, share_layer=False, disable_pemb=disable_std_pemb_encoder, model_name="model.encoder", **kwargs):

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindDecoderEmb=bindDecoderEmb, share_layer=share_layer, disable_pemb=disable_pemb, **kwargs)

		self.model_name = model_name
		self.pemb = None if disable_pemb else nn.Parameter(torch.Tensor(xseql, isize).uniform_(- sqrt(2.0 / (isize + xseql)), sqrt(2.0 / (isize + xseql))))
		self.wemb.padding_idx = pad_id
		self.emb_normer = nn.LayerNorm(isize, eps=ieps_ln_default, elementwise_affine=enable_ln_parameters)

		if share_layer:
			_shared_layer = EncoderLayer(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, model_name=model_name)
			self.nets = nn.ModuleList([_shared_layer for i in range(num_layer)])
		else:
			self.nets = nn.ModuleList([EncoderLayer(isize, fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, ahsize=_ahsize, model_name=model_name) for i in range(num_layer)])

	def forward(self, inputs, mask=None, **kwargs):

		seql = inputs.size(1)
		out = self.wemb(inputs)
		if self.pemb is not None:
			out = self.pemb.narrow(0, pemb_start_ind, seql).add(out, alpha=sqrt(out.size(-1)))
		out = self.emb_normer(out)
		if self.drop is not None:
			out = self.drop(out)

		for net in self.nets:
			out = net(out, mask)

		return out if self.out_normer is None else self.out_normer(out)

	def load_plm(self, plm_parameters, model_name=None, **kwargs):

		_model_name = parse_none(model_name, self.model_name)
		with torch_no_grad():
			copy_plm_parameter(self.wemb.weight, plm_parameters, "%s.embed_tokens.weight" % _model_name)
			copy_plm_parameter(self.pemb, plm_parameters, "%s.embed_positions.weight" % _model_name)
			copy_plm_parameter(self.emb_normer.weight, plm_parameters, "%s.layernorm_embedding.weight" % _model_name)
			copy_plm_parameter(self.emb_normer.bias, plm_parameters, "%s.layernorm_embedding.bias" % _model_name)
			copy_plm_parameter(self.out_normer.weight, plm_parameters, "%s.layer_norm.weight" % _model_name)
			copy_plm_parameter(self.out_normer.bias, plm_parameters, "%s.layer_norm.bias" % _model_name)
			for i, net in enumerate(self.nets):
				net.load_plm(plm_parameters, model_name=_model_name, layer_idx=i, **kwargs)

	def fix_init(self):

		super(Encoder, self).fix_init()
		if self.pemb is not None:
			with torch_no_grad():
				_ = sqrt(2.0 / sum(self.pemb.size()))
				self.pemb.uniform_(- _, _)
