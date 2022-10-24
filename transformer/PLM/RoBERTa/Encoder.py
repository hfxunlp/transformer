#encoding: utf-8

from cnfg.vocab.plm.roberta import pad_id, pemb_start_ind
from cnfg.plm.roberta.base import num_type
from cnfg.plm.roberta.ihyp import *

from transformer.PLM.BERT.Encoder import Encoder as EncoderBase

class Encoder(EncoderBase):

	def __init__(self, isize, nwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, num_type=num_type, share_layer=False, model_name="roberta", **kwargs):

		super(Encoder, self).__init__(isize, nwd, num_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindDecoderEmb=bindDecoderEmb, num_type=num_type, share_layer=share_layer, model_name=model_name, **kwargs)

		self.wemb.padding_idx = pad_id

	def forward(self, inputs, token_types=None, mask=None):

		seql = inputs.size(1)
		out = self.drop(self.out_normer(self.pemb.narrow(0, pemb_start_ind, seql) + (self.temb.weight[0] if token_types is None else self.temb(token_types)) + self.wemb(inputs)))

		_mask = inputs.eq(pad_id).unsqueeze(1) if mask is None else mask
		for net in self.nets:
			out = net(out, _mask)

		return out
