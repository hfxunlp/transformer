#encoding: utf-8

from utils.relpos import share_rel_pos_cache
from utils.fmt.base import parse_double_value_tuple

from transformer.Probe.Encoder import Encoder
from transformer.Probe.Decoder import Decoder

from transformer.NMT import NMT as NMTBase

from cnfg.ihyp import *

class NMT(NMTBase):

	def __init__(self, isize, snwd, tnwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, global_emb=False, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=False, forbidden_index=None, num_layer_ana=0):

		super(NMT, self).__init__(isize, snwd, tnwd, num_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, global_emb=global_emb, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindDecoderEmb=bindDecoderEmb, forbidden_index=forbidden_index)

		enc_layer, dec_layer = parse_double_value_tuple(num_layer)

		self.enc = Encoder(isize, snwd, enc_layer, fhsize, dropout, attn_drop, num_head, xseql, ahsize, norm_output, num_layer_ana)

		emb_w = self.enc.wemb.weight if global_emb else None

		self.dec = Decoder(isize, tnwd, dec_layer, fhsize, dropout, attn_drop, emb_w, num_head, xseql, ahsize, norm_output, bindDecoderEmb, forbidden_index)

		if rel_pos_enabled:
			share_rel_pos_cache(self)

	def forward(self, inpute, inputo, mask=None):

		_mask = inpute.eq(0).unsqueeze(1) if mask is None else mask

		ence, ence_layer = self.enc(inpute, _mask)

		return self.dec(ence, inputo, ence_layer, _mask)
