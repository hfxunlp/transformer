#encoding: utf-8

from transformer.PLM.BERT.Encoder import Encoder as EncoderBase
from utils.fmt.parser import parse_none
from utils.plm.base import copy_plm_parameter
from utils.torch.comp import torch_all, torch_no_grad

from cnfg.plm.roberta.base import eliminate_type_emb, num_type
from cnfg.plm.roberta.ihyp import *
from cnfg.vocab.plm.roberta import pad_id, pemb_start_ind

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, num_type=num_type, share_layer=False, model_name="roberta", eliminate_type_emb=eliminate_type_emb, **kwargs):

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindDecoderEmb=bindDecoderEmb, num_type=num_type, share_layer=share_layer, model_name=model_name, **kwargs)

		self.wemb.padding_idx = pad_id
		self.eliminate_type_emb = eliminate_type_emb

	def forward(self, inputs, token_types=None, mask=None, **kwargs):

		seql = inputs.size(1)
		out = None if self.pemb is None else self.pemb.narrow(0, pemb_start_ind, seql)
		if self.temb is not None:
			_ = self.temb.weight[0] if token_types is None else self.temb(token_types)
			out = _ if out is None else (out + _)
		_ = self.wemb(inputs)
		out = _ if out is None else (out + _)
		if self.out_normer is not None:
			out = self.out_normer(out)
		if self.drop is not None:
			out = self.drop(out)

		for net in self.nets:
			out = net(out, mask)

		return out

	def load_plm(self, plm_parameters, model_name=None, **kwargs):

		_model_name = parse_none(model_name, self.model_name)
		with torch_no_grad():
			copy_plm_parameter(self.wemb.weight, plm_parameters, "%s.embeddings.word_embeddings.weight" % _model_name)
			copy_plm_parameter(self.pemb, plm_parameters, "%s.embeddings.position_embeddings.weight" % _model_name)
			_temb_key = "%s.embeddings.token_type_embeddings.weight" % _model_name
			if self.eliminate_type_emb and (self.temb.weight.size(0) == 1):
				_temb_w = plm_parameters[_temb_key]
				if not torch_all(_temb_w.eq(0.0)).item():
					self.wemb.weight.add_(_temb_w)
					self.wemb.weight[pad_id].sub_(_temb_w)
				self.temb = None
			else:
				copy_plm_parameter(self.temb.weight, plm_parameters, _temb_key)
			copy_plm_parameter(self.out_normer.weight, plm_parameters, "%s.embeddings.LayerNorm.weight" % _model_name)
			copy_plm_parameter(self.out_normer.bias, plm_parameters, "%s.embeddings.LayerNorm.bias" % _model_name)
			for i, net in enumerate(self.nets):
				net.load_plm(plm_parameters, model_name=_model_name, layer_idx=i, **kwargs)
