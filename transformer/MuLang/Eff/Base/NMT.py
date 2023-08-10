#encoding: utf-8

from transformer.MuLang.Eff.Base.Decoder import Decoder
from transformer.MuLang.Eff.Base.Encoder import Encoder
from transformer.NMT import NMT as NMTBase
from utils.fmt.parser import parse_double_value_tuple
from utils.relpos.base import share_rel_pos_cache

from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

class NMT(NMTBase):

	def __init__(self, isize, snwd, tnwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, global_emb=False, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, forbidden_index=None, ntask=None, **kwargs):

		enc_layer, dec_layer = parse_double_value_tuple(num_layer)

		super(NMT, self).__init__(isize, snwd, tnwd, (enc_layer, dec_layer,), fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, global_emb=global_emb, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindDecoderEmb=bindDecoderEmb, forbidden_index=None, **kwargs)

		self.enc = Encoder(isize, snwd, enc_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, ntask=ntask)

		if global_emb:
			emb_w = self.enc.wemb.weight
			task_emb_w = self.enc.task_emb.weight
		else:
			emb_w = task_emb_w = None

		self.dec = Decoder(isize, tnwd, dec_layer, fhsize=fhsize, dropout=dropout, attn_drop=attn_drop, act_drop=act_drop, emb_w=emb_w, num_head=num_head, xseql=xseql, ahsize=ahsize, norm_output=norm_output, bindemb=bindDecoderEmb, forbidden_index=forbidden_index, ntask=ntask, task_emb_w=task_emb_w)

		if rel_pos_enabled:
			share_rel_pos_cache(self)

	def forward(self, inpute, inputo, taskid=None, mask=None, **kwargs):

		_mask = inpute.eq(pad_id).unsqueeze(1) if mask is None else mask

		return self.dec(self.enc(inpute, taskid=taskid, mask=_mask), inputo, taskid=taskid, src_pad_mask=_mask)

	def decode(self, inpute, taskid=None, beam_size=1, max_len=None, length_penalty=0.0, **kwargs):

		mask = inpute.eq(pad_id).unsqueeze(1)
		_max_len = (inpute.size(1) + max(64, inpute.size(1) // 4)) if max_len is None else max_len

		return self.dec.decode(self.enc(inpute, taskid=taskid, mask=mask), taskid=taskid, src_pad_mask=mask, beam_size=beam_size, max_len=_max_len, length_penalty=length_penalty)
