#encoding: utf-8

from transformer.NMT import NMT as NMTBase
from transformer.Probe.ReDecoder import Decoder
from utils.fmt.parser import parse_double_value_tuple
from utils.relpos.base import share_rel_pos_cache

from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

class NMT(NMTBase):

	def __init__(self, isize, snwd, tnwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, global_emb=False, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, forbidden_index=None, num_layer_ana=None, **kwargs):

		super(NMT, self).__init__(isize, snwd, tnwd, num_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, global_emb=global_emb, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindDecoderEmb=bindDecoderEmb, forbidden_index=forbidden_index)

		emb_w = self.enc.wemb.weight if global_emb else None

		_, dec_layer = parse_double_value_tuple(num_layer)

		self.dec = Decoder(isize, tnwd, dec_layer, fhsize, dropout, attn_drop, act_drop, emb_w, num_head, xseql, ahsize, norm_output, bindDecoderEmb, forbidden_index, num_layer_ana)

		if num_layer_ana <= 0:
			self.enc = None

		if rel_pos_enabled:
			share_rel_pos_cache(self)

	def forward(self, inpute, inputo, mask=None, **kwargs):

		if self.enc is None:
			return self.dec(None, inputo, None)
		else:
			_mask = inpute.eq(pad_id).unsqueeze(1) if mask is None else mask
			return self.dec(self.enc(inpute, _mask), inputo, _mask)

	def load_base(self, base_nmt):

		if self.enc is not None:
			if hasattr(self.enc, "load_base"):
				self.enc.load_base(base_nmt.enc)
			else:
				self.enc = base_nmt.enc
		self.dec.load_base(base_nmt.dec)
