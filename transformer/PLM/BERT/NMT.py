#encoding: utf-8

from utils.plm.base import set_ln_ieps
from utils.relpos.base import share_rel_pos_cache
from utils.fmt.parser import parse_double_value_tuple
from cnfg.vocab.plm.bert import pad_id

from transformer.PLM.BERT.Encoder import Encoder
from transformer.PLM.BERT.Decoder import Decoder
from transformer.NMT import NMT as NMTBase

from cnfg.plm.bert.ihyp import *

class NMT(NMTBase):

	def __init__(self, isize, snwd, tnwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, global_emb=False, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, forbidden_index=None, model_name="bert"):

		enc_layer, dec_layer = parse_double_value_tuple(num_layer)

		_ahsize = isize if ahsize is None else ahsize
		_fhsize = _ahsize * 4 if fhsize is None else fhsize

		super(NMT, self).__init__(isize, snwd, tnwd, (enc_layer, dec_layer,), fhsize=_fhsize, dropout=dropout, attn_drop=attn_drop, global_emb=global_emb, num_head=num_head, xseql=xseql, ahsize=_ahsize, norm_output=norm_output, bindDecoderEmb=bindDecoderEmb, forbidden_index=forbidden_index)

		self.model_name = model_name
		enc_model_name, dec_model_name = parse_double_value_tuple(self.model_name)
		self.enc = Encoder(isize, snwd, enc_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, model_name=enc_model_name)

		emb_w = self.enc.wemb.weight if global_emb else None

		self.dec = Decoder(isize, tnwd, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, emb_w=emb_w, num_head=num_head, model_name=dec_model_name)#, num_layer=dec_layer, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindemb=bindDecoderEmb, forbidden_index=forbidden_index

		set_ln_ieps(self, ieps_ln_default)
		if rel_pos_enabled:
			share_rel_pos_cache(self)

	def forward(self, inpute, token_types=None, mask=None, word_prediction=False):

		_mask = inpute.eq(pad_id).unsqueeze(1) if mask is None else mask

		return self.dec(self.enc(inpute, token_types=token_types, mask=_mask), word_prediction=word_prediction)

	def load_plm(self, plm_parameters, model_name=None, layer_idx=None):

		_model_name = self.model_name if model_name is None else model_name
		enc_model_name, dec_model_name = parse_double_value_tuple(_model_name)
		self.enc.load_plm(plm_parameters, model_name=enc_model_name, layer_idx=layer_idx)
		self.dec.load_plm(plm_parameters, model_name=dec_model_name, layer_idx=layer_idx)

	def update_classifier(self, *args, **kwargs):

		self.dec.update_classifier(*args, **kwargs)
