#encoding: utf-8

from transformer.NMT import NMT as NMTBase
from transformer.Probe.Decoder import Decoder
from transformer.Probe.Encoder import Encoder
from utils.fmt.parser import parse_double_value_tuple
from utils.relpos.base import share_rel_pos_cache

from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

class NMT(NMTBase):

	def __init__(self, isize, snwd, tnwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, global_emb=False, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, forbidden_index=None, num_layer_ana=0, **kwargs):

		super(NMT, self).__init__(isize, snwd, tnwd, num_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, global_emb=global_emb, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindDecoderEmb=bindDecoderEmb, forbidden_index=forbidden_index)

		enc_layer, dec_layer = parse_double_value_tuple(num_layer)

		self.enc = Encoder(isize, snwd, enc_layer, fhsize, dropout, attn_drop, act_drop, num_head, xseql, ahsize, norm_output, num_layer_ana)

		emb_w = self.enc.wemb.weight if global_emb else None

		self.dec = Decoder(isize, tnwd, dec_layer, fhsize, dropout, attn_drop, act_drop, emb_w, num_head, xseql, ahsize, norm_output, bindDecoderEmb, forbidden_index)

		if rel_pos_enabled:
			share_rel_pos_cache(self)

	def forward(self, inpute, inputo, mask=None, **kwargs):

		_mask = inpute.eq(pad_id).unsqueeze(1) if mask is None else mask

		ence, ence_layer = self.enc(inpute, _mask)

		return self.dec(ence, inputo, ence_layer, _mask)
