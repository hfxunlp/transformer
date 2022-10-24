#encoding: utf-8

import torch
from torch import nn

from utils.plm.base import copy_plm_parameter
#from cnfg.vocab.plm.roberta import pad_id

from transformer.PLM.BERT.Decoder import Decoder as DecoderBase

#from cnfg.plm.roberta.ihyp import *

class Decoder(DecoderBase):

	def __init__(self, isize, nwd, num_layer=None, fhsize=None, dropout=0.0, attn_drop=0.0, emb_w=None, num_head=8, model_name="roberta", **kwargs):

		super(Decoder, self).__init__(isize, nwd, num_layer=num_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, emb_w=emb_w, num_head=num_head, model_name=model_name, **kwargs)

		self.rel_classifier = None

	def load_plm(self, plm_parameters, model_name=None, layer_idx=None):

		_model_name = self.model_name if model_name is None else model_name
		with torch.no_grad():
			copy_plm_parameter(self.ff[0].weight, plm_parameters, "lm_head.dense.weight")
			_bias_key = "lm_head.dense.bias"
			if self.ff[0].bias is None and (_bias_key in plm_parameters):
				self.ff[0].bias = nn.Parameter(torch.zeros(self.ff[0].weight.size(0)))
			if self.ff[0].bias is not None:
				copy_plm_parameter(self.ff[0].bias, plm_parameters, _bias_key)
			copy_plm_parameter(self.ff[2].weight, plm_parameters, "lm_head.layer_norm.weight")
			copy_plm_parameter(self.ff[2].bias, plm_parameters, "lm_head.layer_norm.bias")
			copy_plm_parameter(self.classifier.weight, plm_parameters, "lm_head.decoder.weight")
			copy_plm_parameter(self.classifier.bias, plm_parameters, "lm_head.bias")
			copy_plm_parameter(self.pooler[0].weight, plm_parameters, "%s.pooler.dense.weight" % _model_name)
			copy_plm_parameter(self.pooler[0].bias, plm_parameters, "%s.pooler.dense.bias" % _model_name)
