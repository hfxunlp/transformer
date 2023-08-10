#encoding: utf-8

from transformer.APE.Decoder import Decoder
from transformer.APE.Encoder import Encoder
from transformer.NMT import NMT as NMTBase
from utils.fmt.parser import parse_double_value_tuple
from utils.relpos.base import share_rel_pos_cache

from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

class NMT(NMTBase):

	def __init__(self, isize, snwd, tnwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, global_emb=False, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, forbidden_index=None, **kwargs):

		enc_layer, dec_layer = parse_double_value_tuple(num_layer)

		super(NMT, self).__init__(isize, snwd, tnwd, (enc_layer, dec_layer,), fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, global_emb=global_emb, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindDecoderEmb=bindDecoderEmb, forbidden_index=forbidden_index)

		self.enc = Encoder(isize, (snwd, tnwd,), enc_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, global_emb=global_emb)

		emb_w = self.enc.tgt_enc.wemb.weight if global_emb else None

		self.dec = Decoder(isize, tnwd, dec_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindemb=bindDecoderEmb, forbidden_index=forbidden_index)

		if rel_pos_enabled:
			share_rel_pos_cache(self)

	def forward(self, inpute, inputm, inputo, src_mask=None, mt_mask=None, **kwargs):

		_src_mask = inpute.eq(pad_id).unsqueeze(1) if src_mask is None else src_mask
		_mt_mask = inputm.eq(pad_id).unsqueeze(1) if mt_mask is None else mt_mask

		enc_src, enc_mt = self.enc(inpute, inputm, _src_mask, _mt_mask)

		return self.dec(enc_src, enc_mt, inputo, _src_mask, _mt_mask)

	def decode(self, inpute, inputm, beam_size=1, max_len=None, length_penalty=0.0, **kwargs):

		src_mask = inpute.eq(pad_id).unsqueeze(1)
		mt_mask = inputm.eq(pad_id).unsqueeze(1)

		_max_len = (inpute.size(1) + max(64, inpute.size(1) // 4)) if max_len is None else max_len

		enc_src, enc_mt = self.enc(inpute, inputm, src_mask, mt_mask)

		return self.dec.decode(enc_src, enc_mt, src_mask, mt_mask, beam_size, _max_len, length_penalty)
