#encoding: utf-8

from torch import nn

from utils.relpos import share_rel_pos_cache
from utils.fmt.base import parse_double_value_tuple

from transformer.APE.Encoder import Encoder
from transformer.APE.Decoder import Decoder

from cnfg.ihyp import *

class NMT(nn.Module):

	def __init__(self, isize, snwd, tnwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, global_emb=False, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=False, forbidden_index=None):

		super(NMT, self).__init__()

		enc_layer, dec_layer = parse_double_value_tuple(num_layer)

		self.enc = Encoder(isize, (snwd, tnwd,), enc_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, global_emb=global_emb)

		emb_w = self.enc.tgt_enc.wemb.weight if global_emb else None

		self.dec = Decoder(isize, tnwd, dec_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindemb=bindDecoderEmb, forbidden_index=forbidden_index)

		if rel_pos_enabled:
			share_rel_pos_cache(self)

	def forward(self, inpute, inputm, inputo, src_mask=None, mt_mask=None):

		_src_mask = inpute.eq(0).unsqueeze(1) if src_mask is None else src_mask
		_mt_mask = inputm.eq(0).unsqueeze(1) if mt_mask is None else mt_mask

		enc_src, enc_mt = self.enc(inpute, inputm, _src_mask, _mt_mask)

		return self.dec(enc_src, enc_mt, inputo, _src_mask, _mt_mask)

	def decode(self, inpute, inputm, beam_size=1, max_len=None, length_penalty=0.0):

		src_mask = inpute.eq(0).unsqueeze(1)
		mt_mask = inputm.eq(0).unsqueeze(1)

		_max_len = inpute.size(1) + max(64, inpute.size(1) // 4) if max_len is None else max_len

		enc_src, enc_mt = self.enc(inpute, inputm, src_mask, mt_mask)

		return self.dec.decode(enc_src, enc_mt, src_mask, mt_mask, beam_size, _max_len, length_penalty)
