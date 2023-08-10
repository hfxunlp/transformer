#encoding: utf-8

from torch import nn

from transformer.Doc.Para.Base.Decoder import Decoder
from transformer.Doc.Para.Base.Encoder import Encoder
from utils.fmt.parser import parse_double_value_tuple
from utils.relpos.base import share_rel_pos_cache

from cnfg.ihyp import *
from cnfg.vocab.base import pad_id

class NMT(nn.Module):

	def __init__(self, isize, snwd, tnwd, num_layer, fhsize=None, dropout=0.0, attn_drop=0.0, act_drop=None, global_emb=False, num_head=8, xseql=cache_len_default, ahsize=None, norm_output=True, bindDecoderEmb=True, forbidden_index=None, nprev_context=2, num_layer_context=1, **kwargs):

		super(NMT, self).__init__()

		enc_layer, dec_layer = parse_double_value_tuple(num_layer)

		self.enc = Encoder(isize, snwd, enc_layer, fhsize, dropout, attn_drop, act_drop, num_head, xseql, ahsize, norm_output, nprev_context, num_layer_context)

		emb_w = self.enc.enc.wemb.weight if global_emb else None

		self.dec = Decoder(isize, tnwd, dec_layer, fhsize, dropout, attn_drop, act_drop, emb_w, num_head, xseql, ahsize, norm_output, bindDecoderEmb, forbidden_index, nprev_context)

		if rel_pos_enabled:
			share_rel_pos_cache(self)

	def forward(self, inpute, inputo, inputc, mask=None, context_mask=None, **kwargs):

		_mask = inpute.eq(pad_id).unsqueeze(1) if mask is None else mask
		_context_mask = inputc.eq(pad_id).unsqueeze(1) if context_mask is None else context_mask
		ence, contexts, context_masks = self.enc(inpute, inputc, _mask, _context_mask)

		return self.dec(ence, inputo, contexts, _mask, context_masks)

	def decode(self, inpute, inputc, beam_size=1, max_len=None, length_penalty=0.0, **kwargs):

		mask = inpute.eq(pad_id).unsqueeze(1)
		context_mask = inputc.eq(pad_id).unsqueeze(1)

		bsize, nsent, seql = inpute.size()
		_max_len = (seql + max(64, seql // 4)) if max_len is None else max_len
		ence, contexts, context_masks = self.enc(inpute, inputc, mask, context_mask)

		return self.dec.decode(ence, contexts, mask.view(bsize * nsent, 1, seql), context_masks, beam_size, _max_len, length_penalty)

	def load_base(self, base_nmt):

		self.enc.load_base(base_nmt.enc)
		self.dec.load_base(base_nmt.dec)
