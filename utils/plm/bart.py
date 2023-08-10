#encoding: utf-8

import torch
from torch import nn

from modules.dropout import Dropout
from utils.fmt.parser import parse_none
from utils.plm.base import copy_plm_parameter
from utils.torch.comp import torch_no_grad

def load_plm_encoder_layer(layer, plm_parameters, model_name=None, layer_idx=None, **kwargs):

	_model_name = parse_none(model_name, layer.model_name)
	with torch_no_grad():
		copy_plm_parameter(layer.attn.net.adaptor.weight, plm_parameters, ["%s.layers.%d.self_attn.q_proj.weight" % (_model_name, layer_idx,), "%s.layers.%d.self_attn.k_proj.weight" % (_model_name, layer_idx,), "%s.layers.%d.self_attn.v_proj.weight" % (_model_name, layer_idx,)], func=torch.cat, func_kwargs={"dim": 0})
		_bias_key = "%s.layers.%d.self_attn.q_proj.bias" % (_model_name, layer_idx,)
		if (layer.attn.net.adaptor.bias is None) and (_bias_key in plm_parameters):
			layer.attn.net.adaptor.bias = nn.Parameter(torch.zeros(layer.attn.net.adaptor.weight.size(0)))
		if layer.attn.net.adaptor.bias is not None:
			copy_plm_parameter(layer.attn.net.adaptor.bias, plm_parameters, [_bias_key, "%s.layers.%d.self_attn.k_proj.bias" % (_model_name, layer_idx,), "%s.layers.%d.self_attn.v_proj.bias" % (_model_name, layer_idx,)], func=torch.cat, func_kwargs={"dim": 0})
		copy_plm_parameter(layer.attn.net.outer.weight, plm_parameters, "%s.layers.%d.self_attn.out_proj.weight" % (_model_name, layer_idx,))
		_bias_key = "%s.layers.%d.self_attn.out_proj.bias" % (_model_name, layer_idx,)
		if (layer.attn.net.outer.bias is None) and (_bias_key in plm_parameters):
			layer.attn.net.outer.bias = nn.Parameter(torch.zeros(layer.attn.net.outer.weight.size(0)))
		if layer.attn.net.outer.bias is not None:
			copy_plm_parameter(layer.attn.net.outer.bias, plm_parameters, _bias_key)
		copy_plm_parameter(layer.attn.normer.weight, plm_parameters, "%s.layers.%d.self_attn_layer_norm.weight" % (_model_name, layer_idx,))
		copy_plm_parameter(layer.attn.normer.bias, plm_parameters, "%s.layers.%d.self_attn_layer_norm.bias" % (_model_name, layer_idx,))
		copy_plm_parameter(layer.ff.net[0].weight, plm_parameters, "%s.layers.%d.fc1.weight" % (_model_name, layer_idx,))
		copy_plm_parameter(layer.ff.net[0].bias, plm_parameters, "%s.layers.%d.fc1.bias" % (_model_name, layer_idx,))
		_l = layer.ff.net[-2] if isinstance(layer.ff.net[-1], Dropout) else layer.ff.net[-1]
		copy_plm_parameter(_l.weight, plm_parameters, "%s.layers.%d.fc2.weight" % (_model_name, layer_idx,))
		_bias_key = "%s.layers.%d.fc2.bias" % (_model_name, layer_idx,)
		if (_l.bias is None) and (_bias_key in plm_parameters):
			_l.bias = nn.Parameter(torch.zeros(_l.weight.size(0)))
		if _l.bias is not None:
			copy_plm_parameter(_l.bias, plm_parameters, _bias_key)
		copy_plm_parameter(layer.ff.normer.weight, plm_parameters, "%s.layers.%d.final_layer_norm.weight" % (_model_name, layer_idx,))
		copy_plm_parameter(layer.ff.normer.bias, plm_parameters, "%s.layers.%d.final_layer_norm.bias" % (_model_name, layer_idx,))

def load_plm_decoder_layer(layer, plm_parameters, model_name=None, layer_idx=None, **kwargs):

	_model_name = parse_none(model_name, layer.model_name)
	with torch_no_grad():
		copy_plm_parameter(layer.self_attn.net.adaptor.weight, plm_parameters, ["%s.layers.%d.self_attn.q_proj.weight" % (_model_name, layer_idx,), "%s.layers.%d.self_attn.k_proj.weight" % (_model_name, layer_idx,), "%s.layers.%d.self_attn.v_proj.weight" % (_model_name, layer_idx,)], func=torch.cat, func_kwargs={"dim": 0})
		_bias_key = "%s.layers.%d.self_attn.q_proj.bias" % (_model_name, layer_idx,)
		if (layer.self_attn.net.adaptor.bias is None) and (_bias_key in plm_parameters):
			layer.self_attn.net.adaptor.bias = nn.Parameter(torch.zeros(layer.self_attn.net.adaptor.weight.size(0)))
		if layer.self_attn.net.adaptor.bias is not None:
			copy_plm_parameter(layer.self_attn.net.adaptor.bias, plm_parameters, [_bias_key, "%s.layers.%d.self_attn.k_proj.bias" % (_model_name, layer_idx,), "%s.layers.%d.self_attn.v_proj.bias" % (_model_name, layer_idx,)], func=torch.cat, func_kwargs={"dim": 0})
		copy_plm_parameter(layer.self_attn.net.outer.weight, plm_parameters, "%s.layers.%d.self_attn.out_proj.weight" % (_model_name, layer_idx,))
		_bias_key = "%s.layers.%d.self_attn.out_proj.bias" % (_model_name, layer_idx,)
		if (layer.self_attn.net.outer.bias is None) and (_bias_key in plm_parameters):
			layer.self_attn.net.outer.bias = nn.Parameter(torch.zeros(layer.self_attn.net.outer.weight.size(0)))
		if layer.self_attn.net.outer.bias is not None:
			copy_plm_parameter(layer.self_attn.net.outer.bias, plm_parameters, _bias_key)
		copy_plm_parameter(layer.self_attn.normer.weight, plm_parameters, "%s.layers.%d.self_attn_layer_norm.weight" % (_model_name, layer_idx,))
		copy_plm_parameter(layer.self_attn.normer.bias, plm_parameters, "%s.layers.%d.self_attn_layer_norm.bias" % (_model_name, layer_idx,))
		copy_plm_parameter(layer.cross_attn.net.query_adaptor.weight, plm_parameters, "%s.layers.%d.encoder_attn.q_proj.weight" % (_model_name, layer_idx,))
		_bias_key = "%s.layers.%d.encoder_attn.q_proj.bias" % (_model_name, layer_idx,)
		if (layer.cross_attn.net.query_adaptor.bias is None) and (_bias_key in plm_parameters):
			layer.cross_attn.net.query_adaptor.bias = nn.Parameter(torch.zeros(layer.cross_attn.net.query_adaptor.weight.size(0)))
		if layer.cross_attn.net.query_adaptor.bias is not None:
			copy_plm_parameter(layer.cross_attn.net.query_adaptor.bias, plm_parameters, _bias_key)
		copy_plm_parameter(layer.cross_attn.net.kv_adaptor.weight, plm_parameters, ["%s.layers.%d.encoder_attn.k_proj.weight" % (_model_name, layer_idx,), "%s.layers.%d.encoder_attn.v_proj.weight" % (_model_name, layer_idx,)], func=torch.cat, func_kwargs={"dim": 0})
		_bias_key = "%s.layers.%d.encoder_attn.k_proj.bias" % (_model_name, layer_idx,)
		if (layer.cross_attn.net.kv_adaptor.bias is None) and (_bias_key in plm_parameters):
			layer.cross_attn.net.kv_adaptor.bias = nn.Parameter(torch.zeros(layer.cross_attn.net.kv_adaptor.weight.size(0)))
		if layer.cross_attn.net.kv_adaptor.bias is not None:
			copy_plm_parameter(layer.cross_attn.net.kv_adaptor.bias, plm_parameters, [_bias_key, "%s.layers.%d.encoder_attn.v_proj.bias" % (_model_name, layer_idx,)], func=torch.cat, func_kwargs={"dim": 0})
		copy_plm_parameter(layer.cross_attn.net.outer.weight, plm_parameters, "%s.layers.%d.encoder_attn.out_proj.weight" % (_model_name, layer_idx,))
		_bias_key = "%s.layers.%d.encoder_attn.out_proj.bias" % (_model_name, layer_idx,)
		if (layer.cross_attn.net.outer.bias is None) and (_bias_key in plm_parameters):
			layer.cross_attn.net.outer.bias = nn.Parameter(torch.zeros(layer.cross_attn.net.outer.weight.size(0)))
		if layer.cross_attn.net.outer.bias is not None:
			copy_plm_parameter(layer.cross_attn.net.outer.bias, plm_parameters, _bias_key)
		copy_plm_parameter(layer.cross_attn.normer.weight, plm_parameters, "%s.layers.%d.encoder_attn_layer_norm.weight" % (_model_name, layer_idx,))
		copy_plm_parameter(layer.cross_attn.normer.bias, plm_parameters, "%s.layers.%d.encoder_attn_layer_norm.bias" % (_model_name, layer_idx,))
		copy_plm_parameter(layer.ff.net[0].weight, plm_parameters, "%s.layers.%d.fc1.weight" % (_model_name, layer_idx,))
		copy_plm_parameter(layer.ff.net[0].bias, plm_parameters, "%s.layers.%d.fc1.bias" % (_model_name, layer_idx,))
		_l = layer.ff.net[-2] if isinstance(layer.ff.net[-1], Dropout) else layer.ff.net[-1]
		copy_plm_parameter(_l.weight, plm_parameters, "%s.layers.%d.fc2.weight" % (_model_name, layer_idx,))
		_bias_key = "%s.layers.%d.fc2.bias" % (_model_name, layer_idx,)
		if (_l.bias is None) and (_bias_key in plm_parameters):
			_l.bias = nn.Parameter(torch.zeros(_l.weight.size(0)))
		if _l.bias is not None:
			copy_plm_parameter(_l.bias, plm_parameters, _bias_key)
		copy_plm_parameter(layer.ff.normer.weight, plm_parameters, "%s.layers.%d.final_layer_norm.weight" % (_model_name, layer_idx,))
		copy_plm_parameter(layer.ff.normer.bias, plm_parameters, "%s.layers.%d.final_layer_norm.bias" % (_model_name, layer_idx,))
