#encoding: utf-8

from transformer.PLM.MBART.Decoder import Decoder
from transformer.PLM.MBART.Encoder import Encoder
from transformer.PLM.NMT import NMT as NMTBase
from utils.fmt.parser import parse_double_value_tuple, parse_none
from utils.plm.base import set_ln_ieps
from utils.relpos.base import share_rel_pos_cache

from cnfg.plm.mbart.ihyp import *
from cnfg.vocab.plm.mbart import pad_id

class NMT(NMTBase):

	def __init__(self, isize, snwd, tnwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, global_emb=False, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, forbidden_index=None, model_name=("model.encoder", "model.decoder",), **kwargs):

		enc_layer, dec_layer = parse_double_value_tuple(num_layer)

		_ahsize = parse_none(ahsize, isize)
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(NMT, self).__init__(isize, snwd, tnwd, (enc_layer, dec_layer,), fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, global_emb=global_emb, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindDecoderEmb=bindDecoderEmb, forbidden_index=forbidden_index, model_name=model_name, **kwargs)

		self.model_name = model_name
		enc_model_name, dec_model_name = parse_double_value_tuple(self.model_name)
		self.enc = Encoder(isize, snwd, enc_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, model_name=enc_model_name)

		emb_w = self.enc.wemb.weight if global_emb else None

		self.dec = Decoder(isize, tnwd, dec_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindemb=bindDecoderEmb, forbidden_index=forbidden_index, model_name=dec_model_name)

		set_ln_ieps(self, ieps_ln_default)
		if rel_pos_enabled:
			share_rel_pos_cache(self)

	def forward(self, inpute, inputo, mask=None, word_prediction=False, **kwargs):

		_mask = inpute.eq(pad_id).unsqueeze(1) if mask is None else mask

		return self.dec(self.enc(inpute, _mask), inputo, _mask, word_prediction=word_prediction)

	def decode(self, inpute, beam_size=1, max_len=None, length_penalty=0.0, lang_id=None, **kwargs):

		mask = inpute.eq(pad_id).unsqueeze(1)
		_max_len = (inpute.size(1) + max(64, inpute.size(1) // 4)) if max_len is None else max_len

		return self.dec.decode(self.enc(inpute, mask), mask, beam_size, _max_len, length_penalty, lang_id=lang_id)
